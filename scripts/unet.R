
library(torch)
library(torchvision)
library(tidyverse)
library(magick)
library(zeallot)
library(cowplot)

train_dir <- "data/kaggle_3m_train"
valid_dir <- "data/kaggle_3m_valid"

brainseg_dataset <- dataset(
  name = "mushroom_dataset",

  initialize = function(img_dir,
                        augmentation_params = NULL,
                        random_sampling = FALSE) {
    
    self$images <- tibble(
      img = grep(list.files(train_dir,
                            full.names = TRUE,
                            pattern = "tif",
                            recursive = TRUE),
                 pattern = 'mask', invert = TRUE, value = TRUE),
      mask = grep(list.files(train_dir,
                             full.names = TRUE,
                             pattern = "tif",
                             recursive = TRUE),
                  pattern = 'mask',value = TRUE)
    ) 
    self$slice_weights <- self$calc_slice_weights(self$images$mask)
    self$augmentation_params <- augmentation_params
    self$random_sampling <- random_sampling
  },

  .getitem = function(i) {
    
    index <- if(self$random_sampling == TRUE) sample(1:self$.length(), 1, prob = self$slice_weights)
      else i
    
    image <- self$images$img[index] %>%
      image_read() %>%
      transform_to_tensor() %>%
      transform_rgb_to_grayscale() %>%
      torch_unsqueeze(1)
    mask <- self$images$mask[index] %>%
      image_read() %>%
      transform_to_tensor() %>%
      transform_rgb_to_grayscale() %>%
      torch_unsqueeze(1)
    
    image <- self$min_max_scale(image)
    
    if (!is.null(self$augmentation_params)) {
      
      scale_param <- self$augmentation_params[1]
      c(image, mask) %<-% self$resize(image, mask, scale_param)
      
    }
    list(img = image, mask = mask)
  },
  
  .length = function() {
    nrow(self$images)
  },
  
  calc_slice_weights = function(masks) {
    
    weights <- map_dbl(masks, function(m) {
      img <- as.integer(magick::image_data(image_read(m), channels = "gray"))
      sum(img/255)
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
    
    rnd_scale <- runif(1, 1 - scale_param, 1 + scale_param)
    resized_img <- transform_resize(img, size = rnd_scale * img_size)
    resized_mask <- transform_resize(mask, size = rnd_scale * img_size)
    diff <- dim(resized_img)[2] - img_size
    if (diff > 0) {
      top <- ceiling(diff/2) 
      left <- ceiling(diff/2) 
      resized_img <- transform_crop(resized_img, top, left, img_size, img_size)
      resized_mask <- transform_crop(resized_mask, top, left, img_size, img_size)
    } else {
      resized_img <- transform_pad(resized_img,
                                   padding = - c(ceiling(diff/2),
                                                 floor(diff/2),
                                                 ceiling(diff/2),
                                                 floor(diff/2))
      )
      resized_mask <- transform_pad(resized_mask, padding = - c(ceiling(diff/2),
                                                                floor(diff/2),
                                                                ceiling(diff/2),
                                                                floor(diff/2))
      )
    }
    list(resized_img, resized_mask)
  }
)

# flip
# rotate

train_ds <- brainseg_dataset(train_dir, augmentation_params = c(0.05, 15, 0.5), random_sampling = TRUE)
train_ds$.length()
img_and_mask <- train_ds$.getitem(1)
img <- img_and_mask[[1]]
img %>% as.array() %>% .[1, , ] %>% as.raster() %>% plot()

mask <- img_and_mask[[2]] 
mask %>% as.array() %>% .[1, , ] %>% as.raster() %>% plot()



# Plot augmentation -------------------------------------------------------

valid_ds <- brainseg_dataset(valid_dir, augmentation_params = NULL, random_sampling = TRUE)
img_and_mask <- valid_ds$.getitem(1)
img <- img_and_mask[[1]]
mask <- img_and_mask[[2]] 

imgs <- map (1:24, function(i) {
  c(img, mask) %<-% valid_ds$resize(img, mask, 0.5)
  img %>%
    torch_squeeze() %>%
    torch_transpose(2, 1) %>%
    #transform_rotate(angle = 180) %>%
    as.array() %>%
    as_tibble() %>%
    rowid_to_column(var="X") %>%
    gather(key="Y", value="Z", -1) %>%
    mutate(Y=as.numeric(gsub("V","",Y))) %>%
    ggplot(aes(X, Y, fill= Z)) + 
    geom_raster() +
    theme_void() +
    theme(legend.position="none") +
    theme(aspect.ratio = 1)
})

plot_grid(plotlist = imgs, nrow = 4)
  
