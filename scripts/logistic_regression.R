library(torch)
library(glmnet)
library(lbfgs) 


# logistic regression -----------------------------------------------
# Example from: https://cran.r-project.org/web/packages/lbfgs/vignettes/Vignette.pdf

data(Leukemia)
dim(Leukemia$x)


# lbfgs -------------------------------------------------------------------

X <- Leukemia$x
y <- Leukemia$y

X1 <- cbind(1, X)
init <- rep(0, ncol(X1))

# negative log likelihood with ridge penalty
likelihood <- function(par, X, y, prec) {
  Xbeta <- X %*% par
  -(sum(y * Xbeta - log(1 + exp(Xbeta))) - 0.5 * sum(par^2 * prec))
}

gradient <- function(par, X, y, prec) {
  p <-  1/(1 + exp(-X %*% par)) 
  -(crossprod(X,(y - p)) - par * prec)
}

lbfgs.out <- lbfgs(likelihood, gradient, init, invisible = 1, X = X1, y = y, prec = 2)

lbfgs.out$value


# torch -------------------------------------------------------------------

t_X1 <- torch_tensor(X1, device = "cuda")
t_y <- torch_tensor(y, device = "cuda")
t_likelihood <- function(params, X, y, prec) {
  Xbeta <- X$matmul(params)
  -(torch_sum(y * Xbeta - torch_log(1 + torch_exp(Xbeta))) - 0.5 * torch_sum(torch_square(params) * prec))
}

t_params <- torch_tensor(init, requires_grad = TRUE, device = "cuda")
optimizer <- optim_lbfgs(t_params) # 0.5293621
#optimizer <- optim_lbfgs(t_params, tolerance_change = 1e-14) #0.5293609 

num_iterations <- 2

calc_loss <- function() {
  optimizer$zero_grad()
  loss <- t_likelihood(t_params, t_X1, t_y, 2)
  cat("Loss: ", as.numeric(loss$to(device  = "cpu")), "\n")
  loss$backward()
  loss
}

for (i in 1:num_iterations) {
  cat("Starting step: ", i, "\n")
  optimizer$step(calc_loss)
}

