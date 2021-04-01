
library(torch)
library(tidyverse)


# Rosenbrock function -----------------------------------------------------

a <- 1
b <- 5

rosenbrock <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  (a - x1)^2 + b * (x2 - x1^2)^2 
}

df <- expand_grid(x = seq(-2, 2, by = 0.01), y = seq(-2, 2, by = 0.01)) %>%
  rowwise() %>%
  mutate(z = rosenbrock(c(x, y))) %>%
  ungroup()

ggplot(data = df,
       aes(x = x,
           y = y,
           z = z)) + 
  geom_contour_filled(breaks = as.numeric(torch_logspace(-3, 3, steps = 50)),
                      show.legend = FALSE) +
  theme_minimal() +
  scale_fill_viridis_d(direction = -1) +
  theme(aspect.ratio = 1)

fn <- rosenbrock

# Manual ------------------------------------------------------------------

num_iterations <- 1000

# trial and error ...
lr <- 0.1
lr <- 0.001
lr <- 0.01

# the tensor with respect to which gradients will be computed 
# usually, these are the parameters / weights, not the data
# here, the input values are the parameters, as we're trying to find the minimum
params <- torch_tensor(c(-1, 1), requires_grad = TRUE)

# do gradient descent to decrease function value
for (i in 1:num_iterations) {
  
  if (i %% 50 == 0) cat("Iteration: ", i, "\n")
  
  value <- fn(params)
  if (i %% 50 == 0) cat("Value is: ", as.numeric(value), "\n")
  
  # compute gradient of value w.r.t. params
  value$backward()
  if (i %% 50 == 0) cat("Gradient is: ", as.matrix(params$grad), "\n")
  
  # update
  with_no_grad({
    params$sub_(lr * params$grad)
    params$grad$zero_()
  })
  
  if (i %% 50 == 0) cat("After update: Params is: ", as.matrix(params), "\n\n")
}



# Adam --------------------------------------------------------------------

num_iterations <- 100

params <- torch_tensor(c(-1, 1), requires_grad = TRUE)

lr <- 0.001 # default 
lr <- 1

optimizer <- optim_adam(params, lr)

for (i in 1:num_iterations) {
  optimizer$zero_grad()
  value <- fn(params)
  if (i %% 10 == 0) cat(as.numeric(value), "\n")
  value$backward()
  optimizer$step()
}

params
value


# lbfgs -------------------------------------------------------------------

num_iterations <- 6
params <- torch_tensor(c(-1, 1), requires_grad = TRUE)
optimizer <- optim_lbfgs(params)

calc_loss <- function() {
  optimizer$zero_grad()
  value <- fn(params)
  if (i %% 1 == 0) cat("Iteration: ", as.numeric(value), "\n")
  value$backward()
  value
}
  
for (i in 1:num_iterations) {
  cat("Step: ", i, "\n")
  optimizer$step(calc_loss)
}

params


# optim -------------------------------------------------------------------

rosenbrock_gradient <- function(x) { 
  x1 <- x[1]
  x2 <- x[2]
  c(-400 * x1 * (x2 - x1^2) - 2 * (1 - x1),
    200 * (x2 - x1^2))
}

optim(c(-1, 1), fn) # Nelder-Mead

optim(c(-1, 1), fn, rosenbrock_gradient, method = "BFGS")

optim(c(-1, 1), fn, rosenbrock_gradient, method = "CG")



