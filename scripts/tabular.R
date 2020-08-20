# https://archive.ics.uci.edu/ml/datasets/Mushroom


library(torch)
library(tidyverse)


mushroom_data <- read_csv(
  "data/agaricus-lepiota.data",
  col_names = c(
    "poisonous",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-type",
    "ring-number",
    "spore-print-color",
    "population",
    "habitat"
  ),
  col_types = rep("c", 23) %>% paste(collapse = "")
)

train_indices <- sample(1:nrow(mushroom_data), size = floor(0.8 * nrow(mushroom_data)))
valid_indices <- setdiff(1:nrow(mushroom_data), train_indices)

mushroom_dataset <- dataset(
  name = "mushroom_dataset",

  initialize = function(indices) {
    self$data <- self$prepare_mushroom_data(mushroom_data[indices, ])
  },

  .getitem = function(index) {
    x <- self$data[index, 2:-1]
    y <- self$data[index, 1] - 1

    list(x, y)
  },

  .length = function() {
    self$data$size()[[1]]
  },

  prepare_mushroom_data = function(input) {
    input <- input %>%
      mutate_all(compose(as.integer, as.factor)) %>%
      as.matrix()

    torch_tensor(input)$to(torch_long())
  }
)

train_ds <- mushroom_dataset(train_indices)
train_ds$.length()
train_ds$.getitem(1)

train_dl <- train_ds %>% dataloader(batch_size = 256, shuffle = TRUE)
train_dl$.length()

iter <- train_dl$.iter()
b <- iter$.next()
b

valid_ds <- mushroom_dataset(valid_indices)
valid_ds$.length()

valid_dl <- valid_ds %>% dataloader(batch_size = 256, shuffle = FALSE)
valid_dl$.length()

net <- nn_module(
  "mushroom_net",

  initialize = function(cardinalities,
                        fc1_dim,
                        fc2_dim) {
    self$embeddings <- vector(mode = "list", length = length(cardinalities))
    for (i in 1:length(cardinalities)) {
      self$embeddings[[i]] <- nn_embedding(cardinalities[i], ceiling(cardinalities[i]/2))
    }
    self$fc1 <- nn_linear(sum(map(cardinalities, function(x) ceiling(x/2)) %>% unlist()), fc1_dim)
    self$fc2 <-
      nn_linear(fc1_dim, fc2_dim)
    self$output <- nn_linear(fc2_dim, 1)
  },

  forward = function(x) {
    xs <- vector(mode = "list", length = length(self$embeddings))
    for (i in 1:length(self$embeddings)) {
      xs[[i]] <- self$embeddings[[i]](x[ , i])
    }
    self$fc1(torch_cat(xs, dim = 2)) %>%
      nnf_relu() %>%
      self$fc2() %>%
      self$output() %>%
      nnf_sigmoid()
  }
)

cardinalities <- map(mushroom_data[ , 2:ncol(mushroom_data)], compose(nlevels, as.factor)) %>%
  unlist() %>%
  unname()

fc1_dim <- 64
fc2_dim <- 64

model <- net(
  cardinalities,
  fc1_dim,
  fc2_dim
)

optimizer <- optim_sgd(model$parameters, lr = 0.1)

for (epoch in 1:20) {

  model$train()
  train_losses <- c()

  for (b in enumerate(train_dl)) {
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- nnf_binary_cross_entropy(output, b[[2]]$to(torch_float()))
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
  }

  model$eval()
  valid_losses <- c()

  for (b in enumerate(valid_dl)) {
    output <- model(b[[1]])
    loss <- nnf_binary_cross_entropy(output, b[[2]]$to(torch_float()))
    valid_losses <- c(valid_losses, loss$item())
  }

  cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, mean(train_losses), mean(valid_losses)))
}

model$eval()
test_dl <- valid_ds %>% dataloader(batch_size = valid_ds$.length(), shuffle = FALSE)
iter <- test_dl$.iter()
b <- iter$.next()b
preds <- as_array(model(b[[1]]))
preds <- ifelse(preds > 0.5, 1, 0)
comp_df <- data.frame(preds = preds, y = b[[2]] %>% as_array())
sum(comp_df$preds == comp_df$y)
sum(comp_df$y == 1)
