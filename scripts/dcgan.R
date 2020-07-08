library(torch)


# Data loading ------------------------------------------------------------


dir <- "/tmp"

batch_size <- 128

kmnist <- kmnist_dataset(
    dir,
    download = TRUE,
    transform = function(x) {
        x <- x$to(dtype = torch_float())/256
        x <- 2*(x - 0.5)
        x[newaxis,..]
    }
)
dl <- dataloader(kmnist, batch_size = batch_size, shuffle = TRUE)


# Model -------------------------------------------------------------------


device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

latent_input_size <- 100
image_size <- 28

generator <- nn_module(
    "generator",
    initialize = function() {
        self$main = nn_sequential(
            # nn_conv_transpose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=TRUE, dilation=1, padding_mode='zeros')
            # h_out = (h_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
            # (1 - 1) * 1 - 2 * 0 + 1 * (4 -1 ) + 0 + 1
            # 4 x 4
            nn_conv_transpose2d(latent_input_size, image_size * 4, 4, 1, 0, bias = FALSE),
            nn_batch_norm2d(image_size * 4),
            nn_relu(),
            # 8 * 8
            nn_conv_transpose2d(image_size * 4, image_size * 2, 4, 2, 1, bias = FALSE),
            nn_batch_norm2d(image_size * 2),
            nn_relu(),
            # 16 x 16
            nn_conv_transpose2d(image_size * 2, image_size, 4, 2, 2, bias = FALSE),
            nn_batch_norm2d(image_size),
            nn_relu(),
            # 28 x 28
            nn_conv_transpose2d(image_size, 1, 4, 2, 1, bias = FALSE),
            nn_tanh()
        )
    },
    forward = function(x) {
        self$main(x)
    }
)

gen <- generator()

discriminator <- nn_module(
    "discriminator",
    initialize = function() {
        self$main = nn_sequential(
            # 14 x 14
            nn_conv2d(1, image_size, 4, 2, 1, bias = FALSE),
            nn_leaky_relu(0.2, inplace = TRUE),
            # 7 x 7
            nn_conv2d(image_size, image_size * 2, 4, 2, 1,  bias = FALSE),
            nn_batch_norm2d(image_size * 2),
            nn_leaky_relu(0.2, inplace = TRUE),
            # 3 x 3
            nn_conv2d(image_size * 2, image_size * 4, 4, 2, 1,  bias = FALSE),
            nn_batch_norm2d(image_size * 4),
            nn_leaky_relu(0.2, inplace = TRUE),
            # 1 x 1
            nn_conv2d(image_size * 4, 1, 4, 2, 1,  bias = FALSE),
            nn_sigmoid()
        )
    },
    forward = function(x) {
        self$main(x)
    }
)

disc <- discriminator()

init_weights <- function(m) {
    if (grepl("conv", m$.classes[[1]])) {
        nn_init_normal_(m$weight$data(), 0.0, 0.02)
    } else if (grepl("batch_norm", m$.classes[[1]])) {
        nn_init_normal_(m$weight$data(), 1.0, 0.02)
        nn_init_constant_(m$bias$data(), 0)
    }
}

gen[[1]]$apply(init_weights)

disc[[1]]$apply(init_weights)

gen$to(device = device)
disc$to(device = device)


# Training ----------------------------------------------------------------


criterion <- nn_bce_loss()

learning_rate <- 0.0002

disc_optimizer <- optim_adam(disc$parameters, lr = learning_rate, betas = c(0.5, 0.999))
gen_optimizer <- optim_adam(gen$parameters, lr = learning_rate, betas = c(0.5, 0.999))

fixed_noise <- torch_randn(c(64, latent_input_size, 1, 1), device = device)
num_epochs <- 5

img_list <- vector(mode = "list", length = num_epochs * trunc(dl$.iter()$.length()/50))
gen_losses <- c()
disc_losses <- c()

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

img_num <- 0
for (epoch in 1:num_epochs) {

    batchnum <- 0
    for (b in enumerate(dl)) {

        batchnum <- batchnum + 1

        y_real <- torch_ones(b[[1]]$size()[1], device = device)
        y_fake <- torch_zeros(b[[1]]$size()[1], device = device)

        noise <- torch_randn(b[[1]]$size()[1], latent_input_size, 1, 1, device = device)
        fake <- gen(noise)
        img <- b[[1]]$to(device = device)

        # update discriminator
        disc_loss <- criterion(disc(img), y_real) + criterion(disc(fake$detach()), y_fake)

        disc_optimizer$zero_grad()
        disc_loss$backward()
        disc_optimizer$step()

        # update generator
        gen_loss <- criterion(disc(fake), y_real)

        gen_optimizer$zero_grad()
        gen_loss$backward()
        gen_optimizer$step()

        disc_losses <- c(disc_losses, disc_loss$cpu()$item())
        gen_losses <- c(gen_losses, gen_loss$cpu()$item())

        if (batchnum %% 50 == 0) {
            img_num <- img_num + 1
            cat("Epoch: ", epoch,
                "    batch: ", batchnum,
                "    disc loss: ", as.numeric(disc_loss$cpu()),
                "    gen loss: ", as.numeric(gen_loss$cpu()),
                "\n")
            with_no_grad({
                generated <- gen(fixed_noise)
                grid <- make_grid(normalize(generated))
                img_list[[img_num]] <- as_array(grid$to(device = "cpu"))
            })
        }

    }
}


# Visualize artifacts over time -------------------------------------------


index <- seq(1, length(img_list), length.out = 16)
images <- img_list[index]

par(mfrow = c(4,4), mar = rep(0.2, 4))
rasterize <- function(x) {
    as.raster(x[1, , ])
}
images %>%
    purrr::map(rasterize) %>%
    purrr::iwalk(~{plot(.x)})


