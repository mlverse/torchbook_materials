library(torch)
library(torchvision)
library(torchdatasets)

library(dplyr)
library(pins)
library(ggplot2)

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

train_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    (function(x)
      x$to(device = device)) %>%
    transform_random_resized_crop(size = c(224, 224)) %>%
    transform_color_jitter() %>%
    transform_random_horizontal_flip() %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406),
                        std = c(0.229, 0.224, 0.225))
}

valid_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    (function(x)
      x$to(device = device)) %>%
    transform_resize(256) %>%
    transform_center_crop(224) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406),
                        std = c(0.229, 0.224, 0.225))
}

test_transforms <- valid_transforms

train_ds <-
  bird_species_dataset("data", download = TRUE, transform = train_transforms)
valid_ds <-
  bird_species_dataset("data", split = "valid", transform = valid_transforms)
test_ds <-
  bird_species_dataset("data", split = "test", transform = test_transforms)

train_ds$.length()
valid_ds$.length()
test_ds$.length()

class_names <- train_ds$classes
length(class_names)

batch_size <- 16
train_dl <-
  dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = batch_size)
test_dl <- dataloader(test_ds, batch_size = batch_size)

train_dl$.length()
valid_dl$.length()
test_dl$.length()

batch <- test_dl$.iter()$.next()
batch[[1]]$size()
batch[[2]]$size()

classes <- batch[[2]]
images <- as_array(batch[[1]]$cpu()) %>%
  aperm(perm = c(1, 3, 4, 2))
mean <- c(0.485, 0.456, 0.406)
std <- c(0.229, 0.224, 0.225)
images <- std * images + mean
images <- images * 255
images[images > 255] <- 255
images[images < 0] <- 0

par(mfcol = c(4, 4), mar = rep(1, 4))

images %>%
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[as_array(classes)]) %>%
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk( ~ {
    plot(.x)
    title(.y)
  })

model <- model_resnet18(pretrained = TRUE)
model$parameters %>% purrr::walk(function(param) param$requires_grad_(FALSE))

num_features <- model$fc$in_features

model$fc <- nn_linear(in_features = num_features, out_features = length(class_names))


model <- model$to(device = device)

criterion <- nn_cross_entropy_loss()

optimizer <- optim_sgd(model$parameters, lr = 0.1, momentum = 0.9)

# find initial learning rate
# ported from: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

losses <- c()
log_lrs <- c()

find_lr <-
  function(init_value = 1e-4,
           final_value = 10,
           beta = 0.98) {
    num <- train_dl$.length()
    mult = (final_value / init_value) ^ (1 / num)
    lr <- init_value
    optimizer$param_groups[[1]]$lr <- lr
    avg_loss <- 0
    best_loss <- 0
    batch_num <- 0

    for (b in enumerate(train_dl)) {
      batch_num <- batch_num + 1
      optimizer$zero_grad()
      output <- model(b[[1]]$to(device = device))
      loss <- criterion(output, b[[2]]$to(device = device))

      #Compute the smoothed loss
      avg_loss <- beta * avg_loss + (1 - beta) * loss$item()
      smoothed_loss <- avg_loss / (1 - beta ^ batch_num)
      #Stop if the loss is exploding
      if (batch_num > 1 && smoothed_loss > 4 * best_loss)
        break
      #Record the best loss
      if (smoothed_loss < best_loss || batch_num == 1)
        best_loss <- smoothed_loss

      #Store the values
      losses <<- c(losses, smoothed_loss)
      log_lrs <<- c(log_lrs, (log(lr, 10)))

      loss$backward()
      optimizer$step()

      #Update the lr for the next step
      lr <- lr * mult
      optimizer$param_groups[[1]]$lr <- lr
    }
  }

find_lr()

losses <- readRDS("scripts/losses.rds")
log_lrs <- readRDS("scripts/log_lrs.rds")

df <- data.frame(log_lrs = log_lrs, losses = losses)
ggplot(df, aes(log_lrs, losses)) +
  geom_point(size = 1) +
  theme_classic()

model <- model_resnet18(pretrained = TRUE)
model$parameters %>% purrr::walk(function(param)
  param$requires_grad_(FALSE))

num_features <- model$fc$in_features

model$fc <-
  nn_linear(in_features = num_features,
            out_features = length(class_names))


model <- model$to(device = device)

criterion <- nn_cross_entropy_loss()

optimizer <- optim_sgd(model$parameters, lr = 0.05, momentum = 0.9)

num_epochs <- 10

scheduler <- optimizer %>%
  lr_one_cycle(
    max_lr = 0.05,
    epochs = num_epochs,
    steps_per_epoch = train_dl$.length()
  )

batch_lrs <- c()

train_batch <- function(b) {

  optimizer$zero_grad()
  output <- model(b[[1]])
  loss <- criterion(output, b[[2]]$to(device = device))
  loss$backward()
  optimizer$step()
  scheduler$step()

  batch_lrs <<- c(batch_lrs, optimizer$param_groups[[1]]$lr)
  loss$item()

}

valid_batch <- function(b) {

  output <- model(b[[1]])
  loss <- criterion(output, b[[2]]$to(device = device))
  gc(full = TRUE)
  loss$item()
}

batch_lrs <- c()

for (epoch in 1:num_epochs) {

  model$train()
  train_losses <- c()
  it <- 0

  for (b in enumerate(train_dl)) {
    it <- it + 1
    loss <- train_batch(b)
    train_losses <- c(train_losses, loss)
  }

  model$eval()
  valid_losses <- c()

  for (b in enumerate(valid_dl)) {
    loss <- valid_batch(b)
    valid_losses <- c(valid_losses, loss)
  }

  cat(sprintf("\nLoss at epoch %d: training: %3f, validation: %3f\n", epoch, mean(train_losses), mean(valid_losses)))
}

df <- data.frame(x = 1:length(batch_lrs), y = batch_lrs)
df %>% ggplot(aes(x, y)) + geom_point(size = 1) +
  xlab("batch") +
  ylab("learning rate") +
  ggtitle("Learning rates (complete training)") +
  theme_classic()

model$eval()

model$eval()

test_batch <- function(b) {

  output <- model(b[[1]])
  labels <- b[[2]]$to(device = device)
  loss <- criterion(output, labels)
  test_losses <<- c(test_losses, loss$item())
  # torch_max returns a list, with position 1 containing the values
  # and position 2 containing the respective indices
  predicted <- torch_max(output$data(), dim = 2)[[2]]
  total <<- total + labels$size(1)
  # add number of correct classifications in this batch to the aggregate
  correct <<- correct + (predicted == labels)$sum()$item()
  gc(full = TRUE)

}

test_losses <- c()
total <- 0
correct <- 0

for (b in enumerate(test_dl)) {
  test_batch(b)
}


mean(test_losses)
test_accuracy <-  correct / total
test_accuracy
