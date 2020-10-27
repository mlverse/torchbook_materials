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
) %>%
  select(-`veil-type`)

nrow(mushroom_data)

train_indices <- sample(1:nrow(mushroom_data), size = floor(0.8 * nrow(mushroom_data)))
valid_indices <- setdiff(1:nrow(mushroom_data), train_indices)

mushroom_dataset <- dataset(
  name = "mushroom_dataset",

  initialize = function(indices) {
    data <- self$prepare_mushroom_data(mushroom_data[indices, ])
    self$xcat <- data[[1]][[1]]
    self$xnum <- data[[1]][[2]]
    self$y <- data[[2]]
  },

  .getitem = function(i) {
    xcat <- self$xcat[i, ]
    xnum <- self$xnum[i, ]
    y <- self$y[i, ]
    
    list(x = list(xcat, xnum), y = y)
  },
  
  .length = function() {
    dim(self$y)[1]
  },
  
  prepare_mushroom_data = function(input) {
    
    input <- input %>%
      mutate(across(.fns = as.factor)) 
    
    target_col <- input$poisonous %>% 
      as.integer() %>%
      `-`(1) %>%
      as.matrix()
    
    categorical_cols <- input %>% 
      select(-poisonous) %>%
      select(where(function(x) nlevels(x) != 2)) %>%
      mutate(across(.fns = as.integer)) %>%
      as.matrix()

    numerical_cols <- input %>%
      select(-poisonous) %>%
      select(where(function(x) nlevels(x) == 2)) %>%
      mutate(across(.fns = as.integer)) %>%
      as.matrix()
    
    list(list(torch_tensor(categorical_cols), torch_tensor(numerical_cols)),
         torch_tensor(target_col))

  }
)

        
train_ds <- mushroom_dataset(train_indices)
train_ds$.length()
#train_ds$.getitem(1)

train_dl <- train_ds %>% dataloader(batch_size = 256, shuffle = TRUE)
train_dl$.length()

#iter <- train_dl$.iter()
#b <- iter$.next()
#b

valid_ds <- mushroom_dataset(valid_indices)
valid_ds$.length()

valid_dl <- valid_ds %>% dataloader(batch_size = 256, shuffle = FALSE)
valid_dl$.length()

embedding_module <- nn_module(
  initialize = function(cardinalities) {
    self$embeddings <- vector(mode = "list", length = length(cardinalities))
    for (i in 1:length(cardinalities)) {
      self$embeddings[[i]] <- nn_embedding(cardinalities[i], ceiling(cardinalities[i]/2))
    }
  },
  forward = function(x) {
    embedded <- vector(mode = "list", length = length(self$embeddings))
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[ , i])
    }
    torch_cat(embedded, dim = 2)
  }
)


net <- nn_module(
  "mushroom_net",

  initialize = function(cardinalities,
                        num_numerical,
                        fc1_dim,
                        fc2_dim) {
    self$embedder <- embedding_module(cardinalities)
    self$fc1 <- nn_linear(sum(map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) + num_numerical, fc1_dim)
    self$fc2 <- nn_linear(fc1_dim, fc2_dim)
    self$output <- nn_linear(fc2_dim, 1)
  },

  forward = function(x) {
    xcat <- x[[1]]
    embedded <- self$embedder(xcat)
    all <- torch_cat(list(embedded, x[[2]]$to(dtype = torch_float())), dim = 2)
    all %>% self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      self$output() %>%
      nnf_sigmoid()
  }
)

cardinalities <- map(mushroom_data[ , 2:ncol(mushroom_data)], compose(nlevels, as.factor)) %>%
  keep(function(x) x > 2) %>%
  unlist() %>%
  unname()

num_numerical <- ncol(mushroom_data) - length(cardinalities) - 1

fc1_dim <- 64
fc2_dim <- 64

model <- net(
  cardinalities,
  num_numerical,
  fc1_dim,
  fc2_dim
)

optimizer <- optim_sgd(model$parameters, lr = 0.1)

for (epoch in 1:20) {

  model$train()
  train_losses <- c()
  #i <- 1
  

  for (b in enumerate(train_dl)) {
    optimizer$zero_grad()
    output <- model(b$x)
    loss <- nnf_binary_cross_entropy(output, b$y$to(torch_float()))
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
    #cat(i, " \n")
    #i <- i + 1
  }

  model$eval()
  valid_losses <- c()
  #i <- 1
  cat("starting validation\n")

  for (b in enumerate(valid_dl)) {
    output <- model(b$x)
    loss <- nnf_binary_cross_entropy(output, b$y$to(torch_float()))
    valid_losses <- c(valid_losses, loss$item())
    #cat(i, " \n")
    #i <- i + 1
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
