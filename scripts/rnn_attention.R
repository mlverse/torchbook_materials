library(torch)
library(tidyverse)
library(tsibble)
library(tsibbledata)
library(lubridate)
library(fable)


# datasets -----------------------------------------------------------------
vic_elec_daily <- vic_elec %>%
  select(Time, Demand) %>%
  index_by(Date = date(Time)) %>%
  summarise(
    Demand = sum(Demand) / 1e3) 

n_timesteps <- 7 * 2
n_forecast <- 7 * 2

elec_train <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2012, 2013)) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

elec_valid <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2014)) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

elec_test <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2014), month(Date) %in% 1:4) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

train_mean <- mean(elec_train)
train_sd <- sd(elec_train)

elec_dataset <- dataset(
  name = "elec_dataset",
  
  initialize = function(x, n_timesteps, sample_frac = 1) {
    
    self$n_timesteps <- n_timesteps
    self$x <- torch_tensor((x - train_mean) / train_sd)
    
    n <- length(self$x) - self$n_timesteps - 1
    
    self$starts <- sort(sample.int(
      n = n,
      size = n * sample_frac
    ))
    
  },
  
  .getitem = function(i) {
    
    start <- self$starts[i]
    end <- start + self$n_timesteps - 1
    lag <- 1
    
    list(
      x = self$x[start:end],
      y = self$x[(start+lag):(end+lag)]$squeeze(2)
    )
    
  },
  
  .length = function() {
    length(self$starts) 
  }
)

train_ds <- elec_dataset(elec_train, n_timesteps)
length(train_ds)
train_ds[1]

batch_size <- 32
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
    
    # return outputs for all timesteps, as well as last-timestep states for all layers
    x %>% self$rnn()
    
  }
  
)

attention_module_additive <- nn_module(
  
  initialize = function(hidden_dim, attention_size) {
    
    self$attention <- nn_linear(2 * hidden_dim, attention_size)
    
  },
  
  forward = function(state, encoder_outputs) {
    
    ##################################################################################
    # calculate attention weights 
    #
    # == weight encoder outputs from all timesteps (KEYS) as to their importance for the CURRENT
    #    decoder hidden state (QUERY)
    #
    # this is done through:
    # - multiplexing the current state,
    # - concatenating with encoder outputs,
    # - passing through a linear layer followed by a tanh, and
    # - applying a softmax
    #
    # this is a form of additive attention
    ##################################################################################
    
    # function argument shapes
    # encoder_outputs: (bs, timesteps, hidden_dim)
    # state: (1, bs, hidden_dim)
    
    # multiplex state to allow for concatenation (dimensions 1 and 2 must agree)
    seq_len <- dim(encoder_outputs)[2]
    # resulting shape: (bs, timesteps, hidden_dim)
    state_rep <- state$permute(c(2, 1, 3))$repeat_interleave(seq_len, 2)
    
    # concatenate along feature dimension
    concat <- torch_cat(list(state_rep, encoder_outputs), dim = 3)
    
    # run through linear layer with tanh
    # resulting shape: (bs, timesteps, attention_size)
    scores <- self$attention(concat) %>% 
      torch_tanh()
    
    # sum over attention dimension and normalize
    # resulting shape: (bs, timesteps) 
    attention_weights <- scores %>%
      torch_sum(dim = 3) %>%
      nnf_softmax(dim = 2)
    
    # a normalized score for every source token
    attention_weights
  }
)

attention_module_multiplicative <- nn_module(
  
  initialize = function() {
    
    NULL
    
  },
  
  forward = function(state, encoder_outputs) {
    
    ##################################################################################
    # calculate attention weights 
    #
    # == weight encoder outputs from all timesteps (KEYS) as to their importance for the CURRENT
    #    decoder hidden state (QUERY)
    #
    # this is done through:
    # - calculating the dot products between key and query vectors
    # - dividing by square root of number of features to achieve unit variance
    #
    # this is a form of multiplicative attention
    ##################################################################################
    
    # function argument shapes
    # encoder_outputs: (bs, timesteps, hidden_dim)
    # state: (1, bs, hidden_dim)
    
    # allow for matrix multiplication with encoder_outputs
    state <- state$permute(c(2, 3, 1))
    
    # prepare for scaling by number of features
    d <- torch_tensor(dim(encoder_outputs)[3], dtype = torch_float())
    
    # scaled dot products between state and outputs
    # resulting shape: (bs, timesteps, 1)
    scores <- torch_bmm(encoder_outputs, state) %>%
      torch_div(torch_sqrt(d))
    
    # normalize
    # resulting shape: (bs, timesteps) 
    attention_weights <- scores$squeeze(3) %>%
      nnf_softmax(dim = 2)
    
    # a normalized score for every source token
    attention_weights
  }
)

decoder_module <- nn_module(
  
  initialize = function(type, input_size, hidden_size, attention_type, attention_size = 8, num_layers = 1) {
    
    self$type <- type
    
    self$rnn <- if (self$type == "gru") {
      nn_gru(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        batch_first = TRUE
      )
    } else {
      nn_lstm(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        batch_first = TRUE
      )
    }
    
    self$linear <- nn_linear(2 * hidden_size + 1, 1)
    
    self$attention <- if (attention_type == "multiplicative") attention_module_multiplicative()
    else attention_module_additive(hidden_size, attention_size)
    
  },
  
  weighted_encoder_outputs = function(state, encoder_outputs) {
    
    ##################################################################################
    # perform attention pooling == create current context (variable/vector) 
    #
    # == apply attention weights to encoder outputs (VALUES)
    #
    # this is done through:
    # - getting the attention weights from the attention module and 
    # - batch-multiplying them with the encoder outputs
    ##################################################################################
    
    # encoder_outputs is (bs, timesteps, hidden_dim)
    # state is (1, bs, hidden_dim)
    # resulting shape: (bs * timesteps)
    attention_weights <- self$attention(state, encoder_outputs)
    
    # resulting shape: (bs, 1, seq_len)
    attention_weights <- attention_weights$unsqueeze(2)
    
    # resulting shape: (bs, 1, hidden_size)
    weighted_encoder_outputs <- torch_bmm(attention_weights, encoder_outputs)
    
    weighted_encoder_outputs
    
  },
  
  forward = function(x, state, encoder_outputs) {
    
    ##################################################################################
    # calculate prediction based on input (the last value predicted) as well as
    # current context
    #
    # this is done through:
    # - getting the weighted encoder outputs (context) from self$weighted_encoder_outputs,
    # - concatenating with the input, 
    # - running the result through an RNN, and
    # - feeding the ensemble of RNN output, context, and input through an MLP
    ##################################################################################
    
    # encoder_outputs is (bs, timesteps, hidden_dim)
    # state is (1, bs, hidden_dim)
    
    # resulting shape: (bs, 1, hidden_size)
    context <- self$weighted_encoder_outputs(state, encoder_outputs)
    
    # concatenate input and context
    # NOTE: this repeating is done to compensate for the absence of an embedding module
    # that, in NLP, would give x a higher proportion in the concatenation
    x_rep <- x$repeat_interleave(dim(context)[3], 3) 
    rnn_input <- torch_cat(list(x_rep, context), dim = 3)
    
    # resulting shapes: (bs, 1, hidden_size) and (1, bs, hidden_size)
    rnn_out <- self$rnn(rnn_input, state)
    rnn_output <- rnn_out[[1]]
    next_hidden <- rnn_out[[2]]
    
    mlp_input <- torch_cat(list(rnn_output$squeeze(2), context$squeeze(2), x$squeeze(2)), dim = 2)
    
    output <- self$linear(mlp_input)
    
    # shapes: (bs, 1) and (1, bs, hidden_size)
    list(output, next_hidden)
  }
  
)

seq2seq_module <- nn_module(
  
  initialize = function(type, input_size, hidden_size, attention_type, attention_size, n_forecast, 
                        num_layers = 1, encoder_dropout = 0) {
    
    self$encoder <- encoder_module(type = type, input_size = input_size, hidden_size = hidden_size,
                                   num_layers, encoder_dropout)
    self$decoder <- decoder_module(type = type, input_size = 2 * hidden_size, hidden_size = hidden_size,
                                   attention_type = attention_type, attention_size = attention_size, num_layers)
    self$n_forecast <- n_forecast
    
  },
  
  forward = function(x, y, teacher_forcing_ratio) {
    
    outputs <- torch_zeros(dim(x)[1], self$n_forecast)
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
      input <- if (teacher_forcing == TRUE) y[ , t - 1, drop = FALSE] else pred
      input <- input$unsqueeze(3)
      out <- self$decoder(input, state, encoder_outputs)
      pred <- out[[1]]
      state <- out[[2]]
      outputs[ , t] <- pred$squeeze(2)
      
    }
    
    outputs
  }
  
)


net <- seq2seq_module("gru", input_size = 1, hidden_size = 32, attention_type = "multiplicative",
                      attention_size = 8, n_forecast = n_forecast)

b <- dataloader_make_iter(train_dl) %>% dataloader_next()
net(b$x, b$y, teacher_forcing_ratio = 1)

# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 1000

train_batch <- function(b, teacher_forcing_ratio) {
  
  optimizer$zero_grad()
  output <- net(b$x, b$y, teacher_forcing_ratio)
  target <- b$y
  
  loss <- nnf_mse_loss(output, target[ , 1:(dim(output)[2])])
  loss$backward()
  optimizer$step()
  
  loss$item()
  
}

valid_batch <- function(b, teacher_forcing_ratio = 0) {
  
  output <- net(b$x, b$y, teacher_forcing_ratio)
  target <- b$y
  
  loss <- nnf_mse_loss(output, target[ , 1:(dim(output)[2])])
  
  loss$item()
  
}

for (epoch in 1:num_epochs) {
  
  net$train()
  train_loss <- c()
  
  coro::loop(for (b in train_dl) {
    loss <-train_batch(b, teacher_forcing_ratio = 0.0)
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


# predict ---------------------------------------------------------

net$eval()

test_preds <- vector(mode = "list", length = length(test_dl))

i <- 1

vic_elec_test <- vic_elec_daily %>%
  filter(year(Date) == 2014, month(Date) %in% 1:4)


coro::loop(for (b in test_dl) {
  
  input <- b$x
  output <- net(b$x, b$y, teacher_forcing_ratio = 0)
  preds <- as.numeric(output)
  
  test_preds[[i]] <- preds
  i <<- i + 1
  
})

test_pred1 <- test_preds[[1]]
test_pred1 <- c(rep(NA, n_timesteps), test_pred1, rep(NA, nrow(vic_elec_test) - n_timesteps - n_forecast))

test_pred2 <- test_preds[[21]]
test_pred2 <- c(rep(NA, n_timesteps + 20), test_pred2, rep(NA, nrow(vic_elec_test) - 20 - n_timesteps - n_forecast))

test_pred3 <- test_preds[[41]]
test_pred3 <- c(rep(NA, n_timesteps + 40), test_pred3, rep(NA, nrow(vic_elec_test) - 40 - n_timesteps - n_forecast))

test_pred4 <- test_preds[[61]]
test_pred4 <- c(rep(NA, n_timesteps + 60), test_pred4, rep(NA, nrow(vic_elec_test) - 60 - n_timesteps - n_forecast))

test_pred5 <- test_preds[[81]]
test_pred5 <- c(rep(NA, n_timesteps + 80), test_pred5, rep(NA, nrow(vic_elec_test) - 80 - n_timesteps - n_forecast))


preds_ts <- vic_elec_test %>%
  select(Demand, Date) %>%
  add_column(
    ex_1 = test_pred1 * train_sd + train_mean,
    ex_2 = test_pred2 * train_sd + train_mean,
    ex_3 = test_pred3 * train_sd + train_mean,
    ex_4 = test_pred4 * train_sd + train_mean,
    ex_5 = test_pred5 * train_sd + train_mean) %>%
  pivot_longer(-Date) %>%
  update_tsibble(key = name)


preds_ts %>%
  autoplot() +
  scale_color_hue(h = c(80, 300), l = 70) +
  theme_minimal()

