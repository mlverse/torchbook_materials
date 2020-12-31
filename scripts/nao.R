

# https://crudata.uea.ac.uk/~timo/datapages/naoi.htm
# https://crudata.uea.ac.uk/cru/data/nao/nao.dat

# https://climatedataguide.ucar.edu/climate-data/hurrell-north-atlantic-oscillation-nao-index-station-based

# https://www.ncdc.noaa.gov/teleconnections/nao/

library(torch)
library(dplyr)
library(readr)
library(ggplot2)

# start 3 years later, in 1824
# last valid value is 2020-7
nao <-
  read_table2(
    "data/nao/nao.dat",
    col_names = FALSE,
    na = "-99.99",
    skip = 3
  ) %>%
  select(-X1,-X14) %>%
  as.matrix() %>%
  t() %>%
  as.numeric() %>%
  .[1:(length(.) - 5)] %>%
  tibble(x = seq.Date(
    from = as.Date("1824-01-01"),
    to = as.Date("2020-07-01"),
    by = "months"
  ),
  y = .)

nao %>% ggplot(aes(x, y)) + geom_line(size = 0.2)

nao_train <- nao %>% filter(x <  as.Date("2000-01-01"))
nao_valid <- nao %>% filter(x >=  as.Date("2000-01-01"))

mean(nao_train$y, na.rm = TRUE)
quantile(nao_train$y, na.rm = TRUE)

mean(nao_valid$y, na.rm = TRUE)
quantile(nao_valid$y, na.rm = TRUE)

nao_train[is.na(nao_train)] <- mean(nao_train$y, na.rm = TRUE)
nao_valid[is.na(nao_valid)] <- mean(nao_train$y, na.rm = TRUE)


# dataset -----------------------------------------------------------------

n_timesteps <- 6

nao_dataset <- dataset(
  name = "nao_dataset",
  
  initialize = function(nao, n_timesteps, random_sample = FALSE) {
    self$nao <- nao$y
    self$n_timesteps <- n_timesteps
    self$random_sample <- random_sample
    
  },
  
  .getitem = function(i) {
    if (self$random_sample == TRUE) {
      i <- sample(1:self$.length(), 1)
    }
    
    x <- torch_tensor(self$nao[i:(n_timesteps + i - 1)])$unsqueeze(2)
    y <- torch_tensor(self$nao[(i + 1):(n_timesteps + i)])
    list(x = x, y = y)
  },
  
  .length = function() {
    length(self$nao) - n_timesteps
  }
  
)

train_ds <- nao_dataset(nao_train, n_timesteps)
length(train_ds)
first <- train_ds$.getitem(1)
first$x
first$y

batch_size <- 32
train_dl <- train_ds %>% dataloader(batch_size = batch_size)
length(train_dl)

iter <- train_dl$.iter()
b <- iter$.next()
b

valid_ds <- nao_dataset(nao_valid, n_timesteps)
length(valid_ds)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)
length(valid_dl)


# model -------------------------------------------------------------------


model <- nn_module(
  initialize = function(type, hidden_size) {
    self$rnn <- if (type == "gru") {
      nn_gru(
        input_size = 1,
        hidden_size = hidden_size,
        batch_first = TRUE
      )
      
    } else {
      nn_lstm(
        input_size = 1,
        hidden_size = hidden_size,
        batch_first = TRUE
      )
      
    }
    
    self$output <- nn_linear(hidden_size, 1)
    
  },
  
  forward = function(x) {
    
    x <- self$rnn(x)[[1]]
    x %>% self$output() %>% torch_flatten(start_dim = 2)
    
  }
  
)

#device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
device <- "cpu"

net <- model("gru", 1)

net <- net$to(device = device)


# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 200

train_batch <- function(b) {
  
  optimizer$zero_grad()
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  
  loss <- nnf_mse_loss(output, target)
  
  if (i %% 100 == 0) {
    print(i)
    
    print(loss$item())
    
    print(as.matrix(output))
    print(as.matrix(target))
  }
  
  i <<- i + 1

  loss$backward()
  optimizer$step()
  
  #gc(full = TRUE)
  
  loss$item()
  
}

valid_batch <- function(b) {
  
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  
  loss <- nnf_mse_loss(output, target)
  
  loss$item()
  
}

for (epoch in 1:num_epochs) {
  
  net$train()
  train_loss <- c()
  
  i <<- 1
  
  for (b in enumerate(train_dl)) {
    
   loss <-train_batch(b)
   train_loss <- c(train_loss, loss)
  }
  
  torch_save(net, paste0("model_", epoch, ".pt"))
  
  cat(sprintf("\nEpoch %d, training: loss: %3.3f \n",
              epoch, mean(train_loss)))
  
  net$eval()
  valid_loss <- c()
  
  for (b in enumerate(valid_dl)) {
    
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
    
  }
  
  cat(sprintf("\nEpoch %d, validation: loss: %3.3f \n",
              epoch, mean(valid_loss)))
}





