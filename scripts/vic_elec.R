library(torch)
library(tidyverse)
library(fable)
library(tsibble)
library(feasts)
library(tsibbledata)
library(lubridate)

# https://otexts.com/fpp3/forecasting.html

# https://otexts.com/fpp3/forecasting-regression.html#forecasting-regression

vic_elec_2014 <- vic_elec %>%
  filter(year(Date) == 2014) %>%
  select(-c(Date, Holiday))

vic_elec_2014_keyed <- vic_elec_2014 %>%
  mutate(Demand = scale(Demand), Temperature = scale(Temperature)) %>%
  pivot_longer(-Time) %>%
  update_tsibble(key = name)

vic_elec_2014_keyed %>% autoplot(alpha = 0.6) + 
  scale_colour_manual(values = c("cyan", "violet")) +
  theme_minimal()
vic_elec_2014_keyed %>% filter(month(Time) == 1) %>% 
  autoplot() + 
  scale_colour_manual(values = c("cyan", "violet")) +
  theme_minimal()

vic_elec_2014_keyed %>% filter(month(Time) == 7) %>% 
  autoplot() + 
  scale_colour_manual(values = c("cyan", "violet")) +
  theme_minimal()

# analysis ----------------------------------------------------------------

# STL
cmp <- vic_elec_2014 %>% 
  model(STL(Demand)) %>%
  components()
cmp %>% autoplot()

cmp <- vic_elec_2014 %>% filter(month(Time) == 7) %>%
  #model(STL(Demand ~ season(period = "week") + season(period = "day"))) %>% # == period = 48, period == 7*48
  model(STL(Demand)) %>% 
  components()
cmp %>% autoplot()

cmp <- vic_elec_2014 %>% filter(month(Time) == 1) %>%
  #model(STL(Demand ~ season(period = "week") + season(period = "day"))) %>% # == period = 48, period == 7*48
  model(STL(Demand)) %>% 
  components()
cmp %>% autoplot()

# other features
# rate at which autocorrelations decrease as the lag between pairs of values increases
# > 0.5: long-term positive autocorrelations
# < 0.5: mean-reverting
# 0.5: random walk
vic_elec_2014 %>% features(Demand, coef_hurst) 
vic_elec_2014 %>% features(Demand, feat_spectral) #[0, 1]

# dataset -----------------------------------------------------------------

n_timesteps <- 7 * 24 * 2

elec_train <- vic_elec_2014$Demand %>% as.matrix()

elec_dataset <- dataset(
  name = "elec_dataset",
  
  initialize = function(demand, n_timesteps) {
    self$demand <-demand
    self$n_timesteps <- n_timesteps
    
  },
  
  .getitem = function(i) {
    
    x <- torch_tensor(self$demand[i:(n_timesteps + i - 1)])$unsqueeze(2)
    y <- torch_tensor(self$demand[n_timesteps + i])
    list(x = x, y = y)
  },
  
  .length = function() {
    length(self$demand) - n_timesteps
  }
  
)

train_ds <- elec_dataset(elec_train, n_timesteps)
length(train_ds)
#first <- train_ds$.getitem(1)
#first$x
#first$y

batch_size <- 32
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
length(train_dl)

# iter <- dataloader_make_iter(train_dl)
# b <- iter %>% dataloader_next()
# b

vic_elec_2015 <- vic_elec %>%
  filter(year(Date) == 2014) %>%
  select(-c(Date, Holiday))

elec_valid <- vic_elec_2015$Demand %>% as.matrix()

valid_ds <- elec_dataset(elec_valid, n_timesteps)
#length(valid_ds)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)
length(valid_dl)


# model -------------------------------------------------------------------


model <- nn_module(
  
  initialize = function(type, input_size, hidden_size, num_layers = 1, dropout = 0) {
    
    self$type <- type
    self$num_layers <- num_layers
    
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
    
    self$output <- nn_linear(hidden_size, 1)
    
  },
  
  forward = function(x) {
    
    # hidden state for last layer, t = seq_len 
    x <- self$rnn(x)
    x <- if (self$type == "gru") x[[2]][self$num_layers,  , ] else x[[2]][[1]][self$num_layers,  , ]
    x <- x$squeeze()
    x %>% self$output() 
  }
  
)

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
device <- "cpu"

net <- model("gru", 1, 32)
net <- net$to(device = device)


# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 300

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
  cat(sprintf("\nEpoch %d, training: loss: %3.3f \n", epoch, mean(train_loss)))
  
  net$eval()
  valid_loss <- c()
  
  coro::loop(for (b in valid_dl) {
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
  })
  
  cat(sprintf("\nEpoch %d, validation: loss: %3.3f \n", epoch, mean(valid_loss)))
}


# predict next -----------------------------------------------------------------

net$eval()

preds <- rep(NA, n_timesteps)

train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)

#dl <- train_dl
dl <- valid_dl

#ds <- nao_train
ds <- nao_valid

#cutoff <- "1987-12"
cutoff <- "2015-12"

coro::loop(for (b in dl) {
  output <- net(b$x$to(device = device))
  preds <- c(preds, output %>% as.numeric())
})

preds_ts <- ds %>%
  add_column(preds) %>%
  pivot_longer(-x) %>%
  update_tsibble(key = name)

preds_ts %>%
  filter(x > yearmonth(cutoff)) %>%
  autoplot()
