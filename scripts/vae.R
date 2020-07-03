## TBD ###

# change dir
# use KMNIST
# use batchnorm
# "cuda:0"
# use Adam


library(zeallot)

dir <- "/tmp"

kmnist <- mnist_dataset(
    dir,
    download = TRUE,
    transform = function(x) {
        x <- x$to(dtype = torch_float())/256
        x[newaxis,..]
    }
)
dl <- dataloader(kmnist, batch_size = 128, shuffle = TRUE)

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

image_size <- 28

view <- nn_module(
    "View",
    initialize = function(shape) {
        self$shape <- shape
    },
    forward = function(x) {
        x$view(self$shape)
    }
)

vae <- nn_module(
    "VAE",

    initialize = function(latent_dim) {
        self$latent_dim <- latent_dim
        self$latent_mean <- nn_linear(896, latent_dim)
        self$latent_log_var <- nn_linear(896, latent_dim)

        self$encoder <- nn_sequential(
            nn_conv2d(1, image_size, kernel_size= 3, stride= 2, padding  = 1),
            #nn_batchnorm_2d(image_size),
            nn_leaky_relu(),
            nn_conv2d(image_size, image_size * 2, kernel_size= 3, stride= 2, padding  = 1),
            #nn_batchnorm_2d(image_size * 2),
            nn_leaky_relu(),
            nn_conv2d(image_size * 2, image_size * 4, kernel_size= 3, stride= 2, padding  = 1),
            #nn_batchnorm_2d(image_size * 4),
            nn_leaky_relu(),
            nn_conv2d(image_size * 4, image_size * 8, kernel_size= 3, stride= 2, padding  = 1),
            #nn_batchnorm_2d(image_size * 8),
            nn_leaky_relu()
        )

        self$decoder <- nn_sequential(
            nn_linear(latent_dim, image_size * 8),
            view(c(-1, image_size * 8, 1, 1)),
            nn_conv_transpose2d(image_size * 8, image_size * 4, kernel_size = 4, stride = 1, padding = 0, bias = FALSE),
            #nn_batchnorm_2d(image_size * 4),
            nn_leaky_relu(),
            # 8 * 8
            nn_conv_transpose2d(image_size * 4, image_size * 2, kernel_size = 4, stride = 2, padding = 1, bias = FALSE),
            #nn_batchnorm_2d(image_size * 2),
            nn_leaky_relu(),
            # 16 x 16
            nn_conv_transpose2d(image_size * 2, image_size, kernel_size = 4, stride = 2, padding = 2, bias = FALSE),
            #nn_batchnorm_2d(image_size),
            nn_leaky_relu(),
            # 28 x 28
            nn_conv_transpose2d(image_size, 1, kernel_size = 4, stride = 2, padding = 1, bias = FALSE),
            nn_sigmoid()
        )
    },

    encode = function(x) {
        result <- self$encoder(x) %>%
            torch_flatten(start_dim = 1)
        mean <- self$latent_mean(result)
        log_var <- self$latent_log_var(result)
        list(mean, log_var)
    },

    decode = function(z) {
        self$decoder(z)
    },

    reparameterize = function(mean, logvar) {
        std <- torch_tensor(0.5, device = "cuda") * logvar
        eps <- torch_randn_like(std)
        eps * std + mean
    },

    loss_function = function(reconstruction, input, mean, log_var) {
        reconstruction_loss <- nnf_binary_cross_entropy(reconstruction, input, reduction = "sum")
        kl_loss <- torch_tensor(-0.5, device = "cuda") * torch_sum(torch_tensor(1, device = "cuda") + log_var - mean^2 - log_var$exp())
        loss <- reconstruction_loss + kl_loss
        list(loss, reconstruction_loss, kl_loss)
    },

    forward = function(x) {
        c(mean, log_var) %<-% self$encode(x)
        z <- self$reparameterize(mean, log_var)
        list (self$decode(z), x, mean, log_var)
    },

    sample = function(num_samples, current_device) {
        z <- torch_randn(num_samples, self$latent_dim)
        z <- z$to(device = current_device)
        samples <- self$decode(z)
        samples
    }

)

model <- vae(latent_dim = 2)$to(device = device)

# TBD replace by Adam
optimizer <- optim_sgd(model$parameters, lr = 0.00001)

num_epochs <- 5

img_list <- vector(mode = "list")

for (epoch in 1:num_epochs) {

    batchnum <- 0
    for (b in enumerate(dl)) {

        batchnum <- batchnum + 1
        input <- b[[1]]$to(device = device)
        optimizer$zero_grad()
        c(reconstruction, input, mean, log_var) %<-% model$forward(input)
        c(loss, reconstruction_loss, kl_loss) %<-% model$loss_function(reconstruction, input, mean, log_var)

        if(batchnum %% 50 == 0) {
            cat("Epoch: ", epoch,
                "    batch: ", batchnum,
                "    loss: ", as.numeric(loss$cpu()),
                "    recon loss: ", as.numeric(reconstruction_loss$cpu()),
                "    KL loss: ", as.numeric(kl_loss$cpu()),
                "\n")
            with_no_grad({
                generated <- model$sample(64, device)
                grid <- make_grid(normalize(generated))
                img_list[[epoch]] <- as_array(grid$to(device = "cpu"))
            })

        }
        loss$backward()
        optimizer$step()

    }


}

normalize <- function(x) {
    min = x$min()$item()
    max = x$max()$item()
    x$clamp_(min = min, max = max)
    x$add_(-min)$div_(max - min + 1e-5)
    x
}

# 4D mini-batch Tensor of shape (B x C x H x W)
make_grid <- function(tensor, num_rows = 8, padding = 2, pad_value = 0) {
    nmaps <- tensor$size(0)
    xmaps <- min(num_rows, nmaps)
    ymaps <- ceiling(nmaps/xmaps)
    height <- floor(tensor$size(2) + padding)
    width <- floor(tensor$size(3) + padding)
    num_channels <- tensor$size(1)
    grid <- tensor$new_full(c(num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k <- 0
    for (y in 0:(ymaps - 1)) {
        for (x in 0:(xmaps - 1)) {
            if (k >= nmaps) break
            grid$narrow(
                dim = 1,
                start = torch_tensor(y * height + padding, dtype = torch_int64())$sum(dim = 0),
                length = height - padding)$narrow(
                    dim = 2,
                    start = torch_tensor(x * width + padding, dtype = torch_int64())$sum(dim = 0),
                    length = width - padding)$copy_(tensor[k + 1, , , ])
            k <- k + 1
        }
    }
    grid
}


index <- seq(1, length(img_list), length.out = 16)
# size is 1 x 242 x 242
images <- img_list[index]

par(mfrow = c(4,4), mar = rep(0.2, 4))
rasterize <- function(x) {
    as.raster(x[1, , ])
}
images %>%
    purrr::map(rasterize) %>%
    purrr::iwalk(~{plot(.x)})
