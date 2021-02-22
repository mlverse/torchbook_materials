# https://en.wikipedia.org/wiki/North_Atlantic_oscillation

# https://ore.exeter.ac.uk/repository/handle/10871/34601

# Previous studies have shown that the El Niño–Southern Oscillation can drive interannual variations
# in the NAO [Brönnimann et al., 2007] and hence Atlantic and European winter climate via the
# stratosphere [Bell et al., 2009].Figures 2b and 2c confirm that this teleconnection to the tropical
# Pacific is active in our experiments, with forecasts initialized in El Niño/La Niña conditions in
# November tending to be followed by negative/positive NAO conditions in winter. 

library(torch)
library(tidyverse)
library(fable)
library(tsibble)
library(feasts)

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
  as.vector() %>%
  .[1:(length(.) - 5)] %>%
  tibble(x = seq.Date(
    from = as.Date("1824-01-01"),
    to = as.Date("2020-07-01"),
    by = "months"
  ),
  nao = .) %>%
  mutate(x = yearmonth(x)) %>%
  fill(nao) %>%
  as_tsibble(index = x) 

nao <- nao %>% filter(x >=  yearmonth("1854-01"))
nrow(nao)

enso <- read_table2("data/ONI_NINO34_1854-2020.txt", skip = 9) %>%
  mutate(x = yearmonth(as.Date(paste0(YEAR, "-", `MON/MMM`, "-01")))) %>%
  select(x, enso = NINO34_MEAN) %>%
  filter(x >= yearmonth("1854-01"), x <= yearmonth("2020-07")) %>%
  as_tsibble(index = x) 

nrow(enso)

nao_mean <- mean(nao$nao)
nao_sd <- sd(nao$nao)

enso_mean <- mean(enso$enso)
enso_sd <- sd(enso$enso)

ts <- nao %>% 
  mutate(nao = (nao - nao_mean) / nao_sd) %>%
  add_column(enso = (enso$enso - enso_mean) / enso_sd) 

ts_train <- ts %>% filter(x <  yearmonth("1970-01"))
ts_valid <- ts %>% filter(x >=  yearmonth("1970-01"))


# analysis ----------------------------------------------------------------

ts_valid_ <- ts_valid %>%
  pivot_longer(-x) %>%
  update_tsibble(key = name)

ts_valid_ %>% autoplot(alpha = 0.3) +
  theme_minimal()

ts_valid_ %>% autoplot()

# STL
cmp <- ts_valid_ %>%
  filter(name == "enso") %>%
  model(STL(value)) %>%
  components()
ts_valid_ %>% autoplot()

ts_valid %>%
  filter(name == "enso") %>%
  features(value, feat_stl)

# ACF
ts_valid_ %>%
  filter(name == "enso") %>%
  ACF(value) %>%
  autoplot()

# other features
# rate at which autocorrelations decrease as the lag between pairs of values increases
# > 0.5: long-term positive autocorrelations
# < 0.5: mean-reverting
# 0.5: random walk
ts_valid_ %>%
  filter(name == "enso") %>% features(value, coef_hurst) 
ts_valid_ %>%
  filter(name == "enso") %>% features(value, feat_spectral) #[0, 1]



# dataset -----------------------------------------------------------------

n_timesteps <- 12

nao_dataset <- dataset(
  name = "nao_dataset",
  
  initialize = function(ts, n_timesteps) {
    
    self$ts <- ts[ , 2:3] %>% as.matrix()
    self$n_timesteps <- n_timesteps
    
  },
  
  .getitem = function(i) {

    x <- torch_tensor(self$ts[i:(self$n_timesteps + i - 1), ])
    y <- torch_tensor(self$ts[self$n_timesteps + i, 1])
    list(x = x, y = y)
  },
  
  .length = function() {
    nrow(self$ts) - self$n_timesteps
  }
  
)

train_ds <- nao_dataset(ts_train, n_timesteps)
length(train_ds)
train_ds[2]

batch_size <-32
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)
length(train_dl)

iter <- dataloader_make_iter(train_dl)
b <- iter %>% dataloader_next()
b

valid_ds <- nao_dataset(ts_valid, n_timesteps)
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

net <- model("gru", 2, 128, 2, 0.8)

net <- net$to(device = device)
net
net(b$x$to(device = device))

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

dl <- train_dl
#dl <- valid_dl

ds <- nao_train
#ds <- nao_valid

cutoff <- "1987-12"
#cutoff <- "2015-12"

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
