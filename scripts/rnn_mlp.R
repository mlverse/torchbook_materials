library(torch)
library(tidyverse)
library(tsibble)
library(tsibbledata)
library(lubridate)
library(fable)


# datasets -----------------------------------------------------------------

n_timesteps <- 7 * 24 * 2
n_forecast <- 7 * 24 * 2 # coincidentally identical to n_timesteps

vic_elec_get_year <- function(year, month = NULL) {
  vic_elec %>%
    filter(year(Date) == year, month(Date) == if (is.null(month)) month(Date) else month) %>%
    as_tibble() %>%
    select(Demand)
}

elec_train <- vic_elec_get_year(2012) %>% as.matrix()
elec_valid <- vic_elec_get_year(2013) %>% as.matrix()
elec_test <- vic_elec_get_year(2014, 1) %>% as.matrix()

train_mean <- mean(elec_train)
train_sd <- sd(elec_train)

elec_dataset <- dataset(
  name = "elec_dataset",
  
  initialize = function(x, n_timesteps, n_forecast, sample_frac = 1) {
    
    self$n_timesteps <- n_timesteps
    self$n_forecast <- n_forecast
    self$x <- torch_tensor((x - train_mean) / train_sd)
    
    n <- length(self$x) - self$n_timesteps - self$n_forecast + 1
    
    self$starts <- sort(sample.int(
      n = n,
      size = n * sample_frac
    ))
    
  },
  
  .getitem = function(i) {
    
    start <- self$starts[i]
    end <- start + self$n_timesteps - 1
    pred_length <- self$n_forecast
    
    list(
      x = self$x[start:end],
      y = self$x[(end + 1):(end + pred_length)]$squeeze(2)
    )
    
  },
  
  .length = function() {
    length(self$starts) 
  }
)

train_ds <- elec_dataset(elec_train, n_timesteps, n_forecast, sample_frac = 0.5)
length(train_ds)
train_ds[1]

batch_size <- 32
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
length(train_dl)
b <- dataloader_make_iter(train_dl) %>% dataloader_next()
b

valid_ds <- elec_dataset(elec_valid, n_timesteps, n_forecast, sample_frac = 0.5)
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
    
    self$mlp <- nn_sequential(
      nn_linear(hidden_size, linear_size),
      nn_relu(),
      nn_dropout(linear_dropout),
      nn_linear(linear_size, output_size)
    )
    
  },
  
  forward = function(x) {
    
    x <- self$rnn(x)
    x[[1]][ ,-1, ..] %>% 
      self$mlp()
    
  }
  
)

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
device <- "cpu"

net <- model("gru", input_size = 1, hidden_size = 32, linear_size = 512, output_size = n_forecast,
             linear_dropout = 0.5)
net <- net$to(device = device)
net

net(b$x)
# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 30

train_batch <- function(b) {
  
  optimizer$zero_grad()
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  
  loss <- nnf_mse_loss(output, target)
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
  
  coro::loop(for (b in train_dl) {
    loss <-train_batch(b)
    train_loss <- c(train_loss, loss)
  })
  
  cat(sprintf("\nEpoch %d, training: loss: %3.5f \n", epoch, mean(train_loss)))
  
  net$eval()
  valid_loss <- c()
  
  coro::loop(for (b in valid_dl) {
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
  })
  
  cat(sprintf("\nEpoch %d, validation: loss: %3.5f \n", epoch, mean(valid_loss)))
}


torch_save(net, "model_multistep.pt")

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

vic_elec_jan_2014 <- vic_elec %>%
  filter(year(Date) == 2014, month(Date) == 1)

test_pred1 <- test_preds[[1]]
test_pred1 <- c(rep(NA, n_timesteps), test_pred1, rep(NA, nrow(vic_elec_jan_2014) - n_timesteps - n_forecast))

test_pred2 <- test_preds[[408]]
test_pred2 <- c(rep(NA, n_timesteps + 407), test_pred2, rep(NA, nrow(vic_elec_jan_2014) - 407 - n_timesteps - n_forecast))

test_pred3 <- test_preds[[817]]
test_pred3 <- c(rep(NA, nrow(vic_elec_jan_2014) - n_forecast), test_pred3)


preds_ts <- vic_elec_jan_2014 %>%
  select(Demand) %>%
  add_column(
    mlp_ex_1 = test_pred1 * train_sd + train_mean,
    mlp_ex_2 = test_pred2 * train_sd + train_mean,
    mlp_ex_3 = test_pred3 * train_sd + train_mean) %>%
  pivot_longer(-Time) %>%
  update_tsibble(key = name)


preds_ts %>%
  autoplot() +
  scale_colour_manual(values = c("#08c5d1", "#00353f", "#ffbf66", "#d46f4d")) +
  theme_minimal()

