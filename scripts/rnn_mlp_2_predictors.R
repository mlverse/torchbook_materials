library(torch)
library(tidyverse)
library(tsibble)
library(tsibbledata)
library(lubridate)
library(fable)


# datasets -----------------------------------------------------------------

n_timesteps <- 7 * 24 * 2
n_forecast <- n_timesteps

vic_elec_get_year <- function(year, month = NULL) {
  vic_elec %>%
    filter(year(Date) == year, month(Date) == if (is.null(month)) month(Date) else month) %>%
    as_tibble() %>%
    select(Demand, Temperature)
}

elec_train <- vic_elec_get_year(2012) %>% as.matrix()
elec_valid <- vic_elec_get_year(2013) %>% as.matrix()
elec_test <- vic_elec_get_year(2014, 1) %>% as.matrix()


train_mean_demand <- mean(elec_train[ , 1])
train_sd_demand <- sd(elec_train[ , 1])

train_mean_temp <- mean(elec_train[ , 2])
train_sd_temp <- sd(elec_train[ , 2])

elec_dataset <- dataset(
  name = "elec_dataset",
  
  initialize = function(data, n_timesteps, n_forecast) {
    demand <- (data[ , 1] - train_mean_demand) / train_sd_demand
    temp <- (data[ , 2] - train_mean_temp) / train_sd_temp
    self$data <- cbind(demand, temp)
    self$n_timesteps <- n_timesteps
    self$n_forecast <- n_forecast
    
  },
  
  .getitem = function(i) {
    
    x <- torch_tensor(self$data[i:(self$n_timesteps + i - 1), ])
    y <- torch_tensor(self$data[(self$n_timesteps + i):(self$n_timesteps + i + self$n_forecast - 1), 1])
    list(x = x, y = y)
  },
  
  .length = function() {
    nrow(self$data) - self$n_timesteps - self$n_forecast + 1
  }
  
)

train_ds <- elec_dataset(elec_train, n_timesteps, n_forecast)
length(train_ds)

batch_size <- 32
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
length(train_dl)

valid_ds <- elec_dataset(elec_valid, n_timesteps, n_forecast)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)
length(valid_dl)

test_ds <- elec_dataset(elec_test, n_timesteps, n_forecast)
test_dl <- test_ds %>% dataloader(batch_size = 1)
length(test_dl)

# model -------------------------------------------------------------------

model <- nn_module(
  
  initialize = function(type, input_size, hidden_size, linear_size, output_size,
                        num_layers = 1, dropout = 0, linear_dropout = 0) {
    
    self$type <- type
    self$num_layers <- num_layers
    self$linear_dropout <- linear_dropout
    
    self$rnn <- if (self$type == "gru") {
      nn_gru(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        dropout = dropout,
        batch_first = TRUE
      )
    } else {
      nn_lstm(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        dropout = dropout,
        batch_first = TRUE
      )
    }
    
    self$linear <- nn_linear(hidden_size, linear_size)
    self$output <- nn_linear(linear_size, output_size)
    
  },
  
  forward = function(x) {
    
    x <- self$rnn(x)
    x <- if (self$type == "gru") x[[2]][self$num_layers,  , ] else x[[2]][[1]][self$num_layers,  , ]
    
    if (self$linear_dropout != 0) x <- x %>% nnf_dropout(self$linear_dropout)
    
    x <- x %>% self$linear()
    x <- x %>% torch_relu()
    
    if (self$linear_dropout != 0) x <- x %>% nnf_dropout(self$linear_dropout)
    
    x %>% self$output() 
    
  }
  
)

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
device <- "cpu"

net <- model("gru", input_size = 2, hidden_size = 32, linear_size = 32, output_size = n_forecast,
             linear_dropout = 0)
net <- net$to(device = device)

# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 10

train_batch <- function(b) {
  
  optimizer$zero_grad()
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  
  loss <- nnf_mse_loss(output, target)
  
  if (i %% 11111 == 0) {
    
    print(as.matrix(output$to(device = "cpu")))
    print(as.matrix(target$to(device = "cpu")))
  }
  
  i <<- i + 1
  
  loss$backward()
  optimizer$step()
  
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
  
  coro::loop(for (b in train_dl) {
    loss <-train_batch(b)
    train_loss <- c(train_loss, loss)
  })
  
  if (epoch %% 100 == 0) torch_save(net, paste0("model_", epoch, ".pt"))
  cat(sprintf("\nEpoch %d, training: loss: %3.5f \n", epoch, mean(train_loss)))
  
  net$eval()
  valid_loss <- c()
  
  coro::loop(for (b in valid_dl) {
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
  })
  
  cat(sprintf("\nEpoch %d, validation: loss: %3.5f \n", epoch, mean(valid_loss)))
}

torch_save(net, "model_multivariate.pt")


# predict ---------------------------------------------------------

net$eval()

test_preds <- vector(mode = "list", length = length(test_dl))

i <- 1

coro::loop(for (b in test_dl) {
  
  input <- b$x
  output <- net(input$to(device = device))
  preds <- as.numeric(output)
  
  test_preds[[i]] <- preds
  i <<- i + 1
  
})

test_pred1 <- test_preds[[1]]
test_pred1 <- c(rep(NA, n_timesteps), test_pred1, rep(NA, nrow(vic_elec_jan_2014) - n_timesteps - n_forecast))

test_pred2 <- test_preds[[408]]
test_pred2 <- c(rep(NA, n_timesteps + 407), test_pred2, rep(NA, nrow(vic_elec_jan_2014) - 407 - n_timesteps - n_forecast))

test_pred3 <- test_preds[[817]]
test_pred3 <- c(rep(NA, nrow(vic_elec_jan_2014) - n_forecast), test_pred3)


preds_ts <- vic_elec %>%
  filter(year(Date) == 2014, month(Date) == 1) %>%
  select(Demand) %>%
  add_column(
    iterative_ex_1 = test_pred * train_sd + train_mean,
    iterative_ex_2 = test_pred2 * train_sd + train_mean,
    iterative_ex_3 = test_pred3 * train_sd + train_mean) %>%
  pivot_longer(-Time) %>%
  update_tsibble(key = name)

preds_ts %>%
  autoplot() +
  scale_colour_manual(values = c("#08c5d1", "#00353f", "#ffbf66", "#d46f4d")) +
  theme_minimal()

