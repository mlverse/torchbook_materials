library(torch)
library(glmnet)
library(lbfgs) 
library(microbenchmark)


# Poisson regression -----------------------------------------------
# Example from: https://cran.r-project.org/web/packages/lbfgs/vignettes/Vignette.pdf

set.seed(777)

# lbfgs -------------------------------------------------------------------

N <- 500
#N <- 5000

p <- 20
nzc <- 5
x <- matrix(rnorm(N * p), N, p)
beta <- rnorm(nzc)
f <- x[, seq(nzc)] %*% beta
mu <- exp(f)
y <- rpois(N, mu)
X1 <- cbind(1,x)
init <- rep(0, ncol(X1))

# way 1: glmnet
glmnet_fit <- function() glmnet(x, y, family="poisson", standardize=FALSE)

# way 2: lbfgs
# We choose a value of the regularization parameter from the model fitted with glmnet
# as the OWL-QN penalty coefficient to obtain analogous results using lbfgs:
C <- glmnet_fit()$lambda[25]*nrow(x)
C

likelihood <- function(par, X, y, prec=0) {
  Xbeta <- X %*% par
  -(sum(y * Xbeta - exp(Xbeta)) - .5 * sum(par^2*prec))
}

gradient <- function(par, X, y, prec=0) {
  Xbeta <- X %*% par
  -(crossprod(X, (y - exp(Xbeta))) - par * prec)
}

lbfgs_fit  <- function()
  lbfgs(
    likelihood,
    gradient,
    init,
    X = X1,
    y = y,
    prec = 0,
    invisible = 1,
    orthantwise_c = C,
    linesearch_algorithm = "LBFGS_LINESEARCH_BACKTRACKING",
    orthantwise_start = 1,
    orthantwise_end = ncol(X1)
  )

# torch -------------------------------------------------------------------

num_iterations <- 3

t_X1 <- torch_tensor(X1, device = "cuda")
t_y <- torch_tensor(y, device = "cuda")

t_likelihood <- function(par, X, y, prec=0) {
  Xbeta <- X$matmul(par)
  -(torch_sum(y * Xbeta - torch_exp(Xbeta)) - .5 * torch_sum(torch_square(par) * prec))
}

t_lbfgs <- function(verbose = FALSE) {
  t_params <- torch_tensor(init, requires_grad = TRUE, device = "cuda")
  optimizer <- optim_lbfgs(t_params, lr = 0.0015, line_search_fn = "strong_wolfe") 
  calc_loss <- function() {
    optimizer$zero_grad()
    loss <- t_likelihood(t_params, t_X1, t_y, 0)
    if (verbose) cat("Loss: ", as.numeric(loss$to(device  = "cpu")), "\n")
    loss$backward()
    loss
  }
  for (i in 1:num_iterations) {
    if (verbose) cat("Starting step: ", i, "\n")
    optimizer$step(calc_loss)
  }
}



# compare -----------------------------------------------------------------


lbfgs_fit()
t_lbfgs(TRUE)

b <- microbenchmark(glmnet_fit(), lbfgs_fit(), t_lbfgs(), times = 10L)
ggplot2::autoplot(b)

