library(torch)
library(torchvision)
library(dplyr)

train_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    transform_random_resized_crop(size = c(224, 224)) %>%
    transform_color_jitter() %>%
    transform_random_horizontal_flip() %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

valid_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    transform_resize(256) %>%
    transform_center_crop(224) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

test_transforms <- valid_transforms

target_transform = function(x) {
  x <- torch_tensor(x, dtype = torch_long())
  x$squeeze(1)
}

# https://www.kaggle.com/gpiosenka/100-bird-species/data
data_dir = 'data/bird_species'

train_ds <- image_folder_dataset(file.path(data_dir, "train"),
                                 transform = train_transforms,
                                 target_transform = target_transform)
valid_ds <- image_folder_dataset(file.path(data_dir, "valid"),
                                 transform = valid_transforms,
                                 target_transform = target_transform)
test_ds <-
  image_folder_dataset(file.path(data_dir, "test"),
                       transform = test_transforms,
                       target_transform = target_transform)

train_dl <- dataloader(train_ds, batch_size = 16, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 16)
test_dl <- dataloader(test_ds, batch_size = 16)

train_dl$.length()
valid_dl$.length()
test_dl$.length()

model <- model_resnet18(pretrained = TRUE)
model$parameters %>% purrr::walk(function(param) param$requires_grad <- FALSE)

num_features <- model$fc$in_features
class_names <- train_ds$classes
class_names

b <- test_dl$.iter()$.next()

# TBD
model$fc <- nn_linear(in_features = num_features, out_features = length(class_names))
model$fc$out_feature <- 1

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

model <- model$to(device = device)

criterion <- nn_cross_entropy_loss()

optimizer <- optim_sgd(model$parameters, lr = 0.001, momentum = 0.9)

# TBD
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for (epoch in 1:1) {

  model$train()
  train_losses <- c()

  for (b in enumerate(train_dl)) {
    optimizer$zero_grad()
    output <- model(b[[1]]$to(device = "cuda"))
    loss <- criterion(output, b[[2]]$to(device = "cuda"))
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
    print(loss)
  }
  #scheduler$step()

  model$eval()
  valid_losses <- c()

  for (b in enumerate(valid_dl)) {
    output <- model(b[[1]])
    loss <- criterion(output, b[[2]]$to(device = "cuda"))
    valid_losses <- c(valid_losses, loss$item())
  }

  cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, mean(train_losses), mean(valid_losses)))
}

model$eval()
iter <- test_dl$.iter()
b <- test_dl$.iter()$.next()
preds <- as_array(model(b[[1]]))
preds <- ifelse(preds > 0.5, 1, 0)
comp_df <- data.frame(preds = preds, y = b[[2]] %>% as_array())
sum(comp_df$preds == comp_df$y)


