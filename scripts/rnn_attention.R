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
    select(Demand)
}

elec_train <- vic_elec_get_year(2012) %>% as.matrix()
elec_valid <- vic_elec_get_year(2013) %>% as.matrix()
elec_test <- vic_elec_get_year(2014, 1) %>% as.matrix()

train_mean <- mean(elec_train)
train_sd <- sd(elec_train)

elec_dataset <- dataset(
  name = "elec_dataset",
  
  initialize = function(demand, n_timesteps) {
    self$demand <- (demand - train_mean) / train_sd
    self$n_timesteps <- n_timesteps
    
  },
  
  .getitem = function(i) {
    
    x <- torch_tensor(self$demand[i:(self$n_timesteps + i - 1)])$unsqueeze(2)
    y <- torch_tensor(self$demand[(i + 1):(self$n_timesteps + i)])
    list(x = x, y = y)
  },
  
  .length = function() {
    length(self$demand) - 2 * self$n_timesteps + 1
  }
  
)

train_ds <- elec_dataset(elec_train, n_timesteps)
length(train_ds)

batch_size <- 4
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
length(train_dl)

valid_ds <- elec_dataset(elec_valid, n_timesteps)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)
length(valid_dl)

test_ds <- elec_dataset(elec_test, n_timesteps)
test_dl <- test_ds %>% dataloader(batch_size = 1)
length(test_dl)

# model -------------------------------------------------------------------

encoder_module <- nn_module(
  
  initialize = function(type, input_size, hidden_size, num_layers = 1, dropout = 0) {
    
    self$type <- type
    
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
    
  },
  
  forward = function(x) {
    
    x <- self$rnn(x)
    
    # return outputs for all timesteps, as well as last-timestep states for all layers
    x
    
  }
  
)

attention_module <- nn_module(
  
  initialize = function(hidden_dim, attention_size) {
    self$attention <- nn_linear(2 * hidden_dim, attention_size)
  },
  
  forward = function(state, encoder_outputs) {
    
    ##################################################################################
    # calculate attention weights 
    #
    # == weight encoder outputs from all timesteps as to their importance for the CURRENT
    #    decoder hidden state
    #
    # this is done through:
    # - multiplexing the current state,
    # - concatenating with encoder outputs,
    # - passing through a linear layer followed by a tanh, and
    # - applying a softmax
    #
    # this is a form of additive attention
    ##################################################################################
    
    # encoder_outputs is (bs, timesteps, hidden_dim)
    # state is (1, bs, hidden_dim)
    
    seq_len <- dim(encoder_outputs)[2]
    # (timesteps, bs, hidden_dim)
    state_rep <- state$repeat_interleave(seq_len, 1)
    
    # (timesteps, bs, hidden_dim)
    encoder_outputs <- encoder_outputs$permute(c(2, 1, 3))
    
    # => concatenates, for every batch item and timestep, hidden state from decoder
    # (encoder, initially) and encoder output
    concat <- torch_cat(list(state_rep, encoder_outputs), dim = 3)
    
    # (timesteps, bs, attention_size)
    # tbd: better variable name?
    att <- self$attention(concat) %>% 
      torch_tanh()
    
    # (timesteps, bs) 
    # a score for every source token
    # tbd: better variable name?
    attention <- torch_sum(att, dim = 3) %>%
      nnf_softmax(dim = 1)
  }
)


decoder_module <- nn_module(
  
  initialize = function(type, input_size, hidden_size, attention_size, num_layers = 1, dropout = 0) {
    
    self$type <- type
    
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
    
    self$linear <- nn_linear(2 * hidden_size + 1, 1)
    
    self$attention <- attention_module(hidden_size, attention_size)
    
  },
  
  weighted_encoder_outputs = function(state, encoder_outputs) {
    
    ##################################################################################
    # calculate weighted encoder outputs (a.k.a.context vector) 
    #
    # == weight encoder outputs from all timesteps as to their importance for the CURRENT
    #    decoder hidden state
    #
    # this is done through:
    # - getting the attention weights from the attention module and 
    # - multiplying them with the encoder outputs
    ##################################################################################
    
    # encoder_outputs is (bs, timesteps, hidden_dim)
    # state is (1, bs, hidden_dim)
    # bs * seq_len
    attention_weights <- self$attention(state, encoder_outputs)
    
    # (bs, 1, seq_len)
    attention_weights <- attention_weights$unsqueeze(2)$permute(c(3, 2, 1))

    # (bs, 1, hidden_size)
    weighted_encoder_outputs <- torch_bmm(attention_weights, encoder_outputs)
    
    weighted_encoder_outputs
    
  },
  
  forward = function(x, state, encoder_outputs) {
    
    ##################################################################################
    # calculate prediction based on input (the last value predicted) as well as
    # weighted encoder outputs
    #
    # this is done through:
    # - getting the weighted encoder outputs from self$weighted_encoder_outputs,
    # - concatenating with the input, 
    # - running the result through an RNN, and
    # - feeding the ensemble of RNN output, weighted encoder outputs, and input
    # - through an MLP
    ##################################################################################
    
    # encoder_outputs is (bs, timesteps, hidden_dim)
    # state is (1, bs, hidden_dim)
    
    # (bs, 1, hidden_size)
    weighted_encoder_outputs <- self$weighted_encoder_outputs(state, encoder_outputs)
    
    # concatenate input and score from attention module
    
    # NOTE: this repeating is done to compensate for the absence of an embedding module
    # that would give x a higher proportion in the concatenation
    # TBD vary??
    x_rep <- x$repeat_interleave(dim(weighted_encoder_outputs)[3], 3) 
    rnn_input <- torch_cat(list(x_rep, weighted_encoder_outputs), dim = 3)
    
    # (bs, 1, hidden_size) and (1, bs, hidden_size)
    rnn_out <- self$rnn(rnn_input, state)
    rnn_output <- rnn_out[[1]]
    next_hidden <- rnn_out[[2]]

    mlp_input <- torch_cat(list(rnn_output$squeeze(2), weighted_encoder_outputs$squeeze(2), x$squeeze(2)), dim = 2)
    
    output <- self$linear(mlp_input)

    # (bs, 1) and (1, bs, hidden_size)
    list(output, next_hidden)
  }
  
)

seq2seq_module <- nn_module(
  
  initialize = function(type, input_size, hidden_size, attention_size, n_forecast, teacher_forcing_ratio, num_layers = 1, encoder_dropout = 0) {
    
    self$encoder <- encoder_module(type = type, input_size = input_size, hidden_size = hidden_size, num_layers, encoder_dropout)
    self$decoder <- decoder_module(type = type, input_size = 2 * hidden_size, hidden_size = hidden_size, attention_size = attention_size, num_layers)
    self$n_forecast <- n_forecast
    
  },
  
  forward = function(x, y, teacher_forcing_ratio) {
    
    outputs <- torch_zeros(dim(x)[1], self$n_forecast)$to(device = device)
    encoded <- self$encoder(x)
    encoder_outputs <- encoded[[1]]
    hidden <- encoded[[2]]
    # list of (batch_size, 1), (1, batch_size, hidden_size)
    out <- self$decoder(x[ , n_timesteps, , drop = FALSE], hidden, encoder_outputs)
    # (batch_size, 1)
    pred <- out[[1]]
    # (1, batch_size, hidden_size)
    state <- out[[2]]
    outputs[ , 1] <- pred$squeeze(2)
    
    for (t in 2:self$n_forecast) {
      
      teacher_forcing <- runif(1) < teacher_forcing_ratio
      input <- if (teacher_forcing == TRUE) pred$unsqueeze(3) else y[ , t - 1]
      out <- self$decoder(pred$unsqueeze(3), state, encoder_outputs)
      pred <- out[[1]]
      state <- out[[2]]
      outputs[ , t] <- pred$squeeze(2)
      
    }
    
    outputs
  }
  
)

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
device <- "cpu"

net <- seq2seq_module("gru", input_size = 1, hidden_size = 32, attention_size = 8, n_forecast = n_timesteps, teacher_forcing_ratio = 1)
net <- net$to(device = device)

b <- dataloader_make_iter(train_dl) %>% dataloader_next()
net(b$x, b$y, teacher_forcing_ratio = 1)

# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 1

train_batch <- function(b, teacher_forcing_ratio) {
  
  optimizer$zero_grad()
  output <- net(b$x$to(device = device), b$y$to(device = device), teacher_forcing_ratio)
  target <- b$y$to(device = device)
  
  loss <- nnf_mse_loss(output, target[ , 1:(dim(output)[2])])
  loss$backward()
  optimizer$step()
  
  loss$item()
  
}

valid_batch <- function(b, teacher_forcing_ratio = 0) {
  
  output <- net(b$x$to(device = device), b$y$to(device = device), teacher_forcing_ratio)
  target <- b$y$to(device = device)
  
  loss <- nnf_mse_loss(output, target[ , 1:(dim(output)[2])])
  
  loss$item()
  
}

for (epoch in 1:num_epochs) {
  
  net$train()
  train_loss <- c()
  
  coro::loop(for (b in train_dl) {
    loss <-train_batch(b, teacher_forcing_ratio = 1)
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

torch_save(net, "model_seq2seq.pt")

# predict ---------------------------------------------------------

net$eval()

test_preds <- vector(mode = "list", length = length(test_dl))

coro::loop(for (b in test_dl) {
  
  input <- b$x
  output <- net(input$to(device = device), teacher_forcing_ratio = 0)
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

