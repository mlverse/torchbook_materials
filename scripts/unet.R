
library(torch)
library(torchvision)
library(tidyverse)
library(magick)
library(zeallot)
library(cowplot)



# Dataset -----------------------------------------------------------------

train_dir <- "data/kaggle_3m_train"
valid_dir <- "data/kaggle_3m_valid"

brainseg_dataset <- dataset(
  name = "mushroom_dataset",
  
  initialize = function(img_dir,
                        augmentation_params = NULL,
                        random_sampling = FALSE) {
    self$images <- tibble(
      img = grep(
        list.files(
          train_dir,
          full.names = TRUE,
          pattern = "tif",
          recursive = TRUE
        ),
        pattern = 'mask',
        invert = TRUE,
        value = TRUE
      ),
      mask = grep(
        list.files(
          train_dir,
          full.names = TRUE,
          pattern = "tif",
          recursive = TRUE
        ),
        pattern = 'mask',
        value = TRUE
      )
    )
    self$slice_weights <- self$calc_slice_weights(self$images$mask)
    self$augmentation_params <- augmentation_params
    self$random_sampling <- random_sampling
  },
  
  .getitem = function(i) {
    index <-
      if (self$random_sampling == TRUE)
        sample(1:self$.length(), 1, prob = self$slice_weights)
    else
      i
    
    img <- self$images$img[index] %>%
      image_read() %>%
      transform_to_tensor() %>%
      transform_rgb_to_grayscale() %>%
      torch_unsqueeze(1)
    mask <- self$images$mask[index] %>%
      image_read() %>%
      transform_to_tensor() %>%
      transform_rgb_to_grayscale() %>%
      torch_unsqueeze(1)
    
    img <- self$min_max_scale(img)
    
    if (!is.null(self$augmentation_params)) {
      scale_param <- self$augmentation_params[1]
      c(img, mask) %<-% self$resize(img, mask, scale_param)
      
      rot_param <- self$augmentation_params[2]
      c(img, mask) %<-% self$rotate(img, mask, rot_param)
      
      flip_param <- self$augmentation_params[3]
      c(img, mask) %<-% self$flip(img, mask, flip_param)
      
    }
    list(img = img, mask = mask)
  },
  
  .length = function() {
    nrow(self$images)
  },
  
  calc_slice_weights = function(masks) {
    weights <- map_dbl(masks, function(m) {
      img <-
        as.integer(magick::image_data(image_read(m), channels = "gray"))
      sum(img / 255)
    })
    
    sum_weights <- sum(weights)
    num_weights <- length(weights)
    
    weights <- weights %>% map_dbl(function(w) {
      w <- (w + sum_weights * 0.1 / num_weights) / (sum_weights * 1.1)
    })
    weights
  },
  
  min_max_scale = function(x) {
    min = x$min()$item()
    max = x$max()$item()
    x$clamp_(min = min, max = max)
    x$add_(-min)$div_(max - min + 1e-5)
    x
  },
  
  resize = function(img, mask, scale_param) {
    img_size <- dim(img)[2]
    rnd_scale <- runif(1, 1 - scale_param, 1 + scale_param)
    img <- transform_resize(img, size = rnd_scale * img_size)
    mask <- transform_resize(mask, size = rnd_scale * img_size)
    diff <- dim(img)[2] - img_size
    if (diff > 0) {
      top <- ceiling(diff / 2)
      left <- ceiling(diff / 2)
      img <- transform_crop(img, top, left, img_size, img_size)
      mask <- transform_crop(mask, top, left, img_size, img_size)
    } else {
      img <- transform_pad(img,
                           padding = -c(
                             ceiling(diff / 2),
                             floor(diff / 2),
                             ceiling(diff / 2),
                             floor(diff / 2)
                           ))
      mask <- transform_pad(mask, padding = -c(
        ceiling(diff / 2),
        floor(diff /
                2),
        ceiling(diff /
                  2),
        floor(diff /
                2)
      ))
    }
    list(img, mask)
  },
  
  rotate = function(img, mask, rot_param) {
    rnd_rot <- runif(1, 1 - rot_param, 1 + rot_param)
    img <- transform_rotate(img, angle = rnd_rot)
    mask <- transform_rotate(mask, angle = rnd_rot)
    
    list(img, mask)
  },
  
  flip = function(img, mask, flip_param) {
    rnd_flip <- runif(1)
    if (rnd_flip > flip_param) {
      img <- transform_hflip(img)
      mask <- transform_hflip(mask)
    }
    
    list(img, mask)
  }
)

train_ds <-
  brainseg_dataset(
    train_dir,
    augmentation_params = c(0.5, 180, 0.5),
    random_sampling = TRUE
  )

train_ds$.length()

valid_ds <-
  brainseg_dataset(valid_dir,
                   augmentation_params = NULL,
                   random_sampling = FALSE)


img_and_mask <- train_ds$.getitem(1)
img <- img_and_mask[[1]]
img %>% as.array() %>% .[1, ,] %>% as.raster() %>% plot()

mask <- img_and_mask[[2]]
mask %>% as.array() %>% .[1, ,] %>% as.raster() %>% plot()



# Plot augmentation -------------------------------------------------------

img_and_mask <- valid_ds$.getitem(77)
img <- img_and_mask[[1]]
mask <- img_and_mask[[2]]

imgs <- map (1:24, function(i) {
  c(img, mask) %<-% valid_ds$resize(img, mask, 0.5)
  c(img, mask) %<-% valid_ds$flip(img, mask, 0.5)
  c(img, mask) %<-% valid_ds$rotate(img, mask, 180)
  img %>%
    as.array() %>%
    .[1, ,] %>%
    as_tibble() %>%
    rowid_to_column(var = "X") %>%
    gather(key = "Y", value = "Z",-1) %>%
    mutate(Y = as.numeric(gsub("V", "", Y))) %>%
    ggplot(aes(X, Y, fill = Z)) +
    geom_raster() +
    theme_void() +
    theme(legend.position = "none") +
    theme(aspect.ratio = 1)
})

plot_grid(plotlist = imgs, nrow = 4)



# Load data ---------------------------------------------------------------

batch_size <- 4
train_dl <- dataloader(train_ds, batch_size)
valid_dl <- dataloader(valid_ds, batch_size)



# Model -------------------------------------------------------------------

unet <- nn_module(
  "unet",
  
  initialize = function(channels_in = 1,
                        n_classes = 1,
                        depth = 5,
                        n_filters = 6) {
    
    self$down_path <- nn_module_list()
    
    prev_channels <- channels_in
    for (i in 1:depth) {
      self$down_path$append(down_block(prev_channels, 2 ^ (n_filters + i - 1)))
      prev_channels <- 2 ^ (n_filters + i -1)
    }
    
    self$up_path = nn_module_list()
    for (i in ((depth - 1):1)) {
      self$up_path$append(up_block(prev_channels, 2 ^ (n_filters + i - 1)))
      prev_channels <- 2 ^ (n_filters + i - 1)
    }
    
    self$last = nn_conv2d(prev_channels, n_classes, kernel_size = 1)
  },
  
  forward = function(x) {
    
    blocks <- nn_module_list()
    
    for (i in 1:length(self$down_path)) {
      x <- self$down_path[[i]](x)
      if (i != length(self$down_path)) {
        blocks$append(x)
        x <- nnf_max_pool2d(x, 2)
      }
    }

    for (i in 1:length(self$up_path)) {  
      x <- self$up_path[[i]](x, blocks[[length(blocks) - i + 1]])
    }
    torch_sigmoid(self$last(x))
  }
)

down_block <- nn_module(
  "down_block",
  
  initialize = function(in_size, out_size) {
    self$conv_block <- conv_block(in_size, out_size)
  },
  
  forward = function(x) {
    self$conv_block(x)
  }
)

up_block <- nn_module(
  "up_block",
  initialize = function(in_size, out_size) {
    self$up = nn_conv_transpose2d(in_size,
                                  out_size,
                                  kernel_size = 2,
                                  stride = 2)
    self$conv_block = conv_block(in_size, out_size)
    
  } ,
  forward = function(x, bridge) {
    up <- self$up(x)
    torch_cat(list(up, bridge), 2) %>%
      self$conv_block()
  }
)

conv_block <- nn_module( 
  "conv_block",
  
  initialize = function(in_size, out_size) {
    
    self$block <- nn_module_list()
    self$block$append(nn_conv2d(in_size, out_size, kernel_size = 3, padding = 1))
    self$block$append(nn_relu())
    self$block$append(nn_batch_norm2d(out_size))
    self$block$append(nn_dropout(0.6))
    self$block$append(nn_conv2d(out_size, out_size, kernel_size = 3, padding = 1))
    self$block$append(nn_relu())
    self$block$append(nn_batch_norm2d(out_size))
    self$block$append(nn_dropout(0.6))
  },
  
  forward = function(x){
    for (i in 1:length(self$block)) {
      x <- self$block[[i]](x)
    }
    x
  }
)


device <- torch_device(if(cuda_is_available()) "cuda" else "cpu")
device <- "cpu"
model <- unet(depth = 5)$to(device = device)

b <- train_dl$.iter()$.next()
x <- b[[1]]$to(device = device)
res <- model(x)
res


# Loss, optimizer etc -----------------------------------------------------

calc_dice_loss <- function(y_pred, y_true) {
  smooth <- 1
  y_pred <- y_pred$view(-1)
  y_true <- y_true$view(-1)
  intersection <- (y_pred * y_true)$sum()
  1 - ((2 * intersection + smooth) / (y_pred$sum() + y_true$sum() + smooth))
}

dice_weight <- 0.3

optimizer <- optim_sgd(model$parameters, lr = 0.1, momentum = 0.9)

num_epochs <- 1
scheduler <- lr_one_cycle(optimizer, max_lr = 0.1, steps_per_epoch = train_dl$.length(), epochs = num_epochs)


# Train -------------------------------------------------------------------

train_batch <- function(b) {
  
  optimizer$zero_grad()
  output <- model(b[[1]]$to(device = device))
  target <- b[[2]]$to(device = device)
  
  bce_loss <- nnf_binary_cross_entropy(output, target)
  dice_loss <- calc_dice_loss(output, target)
  loss <-  dice_weight * dice_loss + (1 - dice_weight) * bce_loss
  
  loss$backward()
  optimizer$step()
  scheduler$step()
  
  list(bce_loss$item(), dice_loss$item(), loss$item())
  
}

valid_batch <- function(b) {
  
  output <- model(b[[1]]$to(device = device))
  target <- b[[2]]$to(device = device)
  
  bce_loss <- nnf_binary_cross_entropy(output, target)
  dice_loss <- calc_dice_loss(output, target)
  loss <-  dice_weight * dice_loss + (1 - dice_weight) * bce_loss
  list(bce_loss$item(), dice_loss$item(), loss$item())
}

for (epoch in 1:num_epochs) {
  
  model$train()
  train_bce <- c()
  train_dice <- c()
  train_loss <- c()
  
  for (b in enumerate(train_dl)) {
    c(bce_loss, dice_loss, loss) %<-% train_batch(b)
    train_bce <- c(train_bce, bce_loss)
    train_dice <- c(train_dice, dice_loss)
    train_loss <- c(train_loss, loss)
    cat(sprintf("\nTrain batch: loss:%3f, bce: %3f, dice: %3f\n", train_loss, train_bce, train_dice))
  }
  
  cat(sprintf("\nEpoch %d, training: loss:%3f, bce: %3f, dice: %3f\n",
              epoch, mean(train_loss), mean(train_bce), mean(train_dice)))
  
  model$eval()
  valid_bce <- c()
  valid_dice <- c()
  valid_loss <- c()
  
  for (b in enumerate(valid_dl)) {
    c(bce_loss, dice_loss, loss) %<-% valid_batch(b)
    valid_bce <- c(valid_bce, bce_loss)
    valid_dice <- c(valid_dice, dice_loss)
    valid_loss <- c(valid_loss, loss)
    cat(sprintf("\nTrain batch: loss:%3f, bce: %3f, dice: %3f\n", train_loss, train_bce, train_dice))
    
  }
  
  cat(sprintf("\nEpoch %d, validation: loss:%3f, bce: %3f, dice: %3f\n",
              epoch, mean(valid_loss), mean(valid_bce), mean(valid_dice)))
}


