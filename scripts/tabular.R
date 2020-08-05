# https://doi.org/10.1594/PANGAEA.884141

# The dataset builds upon data from CDP (187 cities, few in developing countries), the Bonn Center for Local Climate Action and Reporting (73 cities, mainly in developing countries), and data collected by Peking University (83 cities in China).

# Scope-1 emissions cover GHGs emitted in the city territory, including emissions from grid-supplied energy produced within cities

library(torch)
library(tidyverse)

co2_data <- read_tsv("../torchbook/data/co2/D_FINAL.tsv") %>%
  select(
    # Five data records related to Scope-1 emissions and three data records related to geo-descriptive attributes were set as common columns in Dfinal to avoid redundancy.
    `Scope-1 source dataset`,

    `Scope-1 GHG emissions [tCO2 or tCO2-eq]`,
    `Year of emission`,

    `Country`,
    `Region`,

    `Population (others)`,
    `Household size (others) [people/household]`,
    `City area (others) [km2]`,

    `HDD 15.5C (clim) [degrees C × days]`,
    `CDD 23C (clim) [degrees C × days]`,

    `Diesel price 2014 (others) [USD/liter]`,
    `Gasoline price 2014 (others) [USD/liter]`
  ) %>%
  rename (
    source = `Scope-1 source dataset`,
    tco2 = `Scope-1 GHG emissions [tCO2 or tCO2-eq]`,
    year = `Year of emission`,
    country = `Country`,
    region = `Region`,
    population = `Population (others)`,
    household_size = `Household size (others) [people/household]`,
    city_area = `City area (others) [km2]`,
    hdd = `HDD 15.5C (clim) [degrees C × days]`,
    cdd = `CDD 23C (clim) [degrees C × days]`,
    diesel_price = `Diesel price 2014 (others) [USD/liter]`,
    gasoline_price = `Gasoline price 2014 (others) [USD/liter]`
  ) %>%
  filter(!is.na(tco2))

co2_dataset <- dataset(
  name = "co2_dataset",

  initialize = function() {
    self$data <- self$prepare_co2_data(co2_data)
  },

  .getitem = function(index) {
    x <- self$data[index, 2:-1]
    y <- self$data[index, 1]

    list(x, y)
  },

  .length = function() {
    self$data$size()[[1]]
  },

  prepare_co2_data = function(co2) {
    input <- cbind(
      self$min_max_scale(co2$tco2),
      as.integer(as.factor(co2$country)) - 1,
      as.integer(as.factor(co2$region)) - 1,
      self$min_max_scale(co2$year),
      self$min_max_scale(self$fill_na(co2$population)),
      self$min_max_scale(self$fill_na(co2$household_size)),
      self$min_max_scale(self$fill_na(co2$city_area)),
      self$min_max_scale(co2$hdd),
      self$min_max_scale(co2$cdd),
      self$min_max_scale(co2$diesel_price),
      self$min_max_scale(co2$gasoline_price)
    )

    torch_tensor(input)
  },

  min_max_scale = function(x) {
    min = min(x)
    max = max(x)
    (x - min) / (max - min + 1e-5)
  },

  fill_na = function(x) {
    fill_value <- mean(na.omit(x))
    x[is.na(x)] <- fill_value
    x
  }
)

ds <- co2_dataset()
ds$.length()
ds$.getitem(1)

dl <- ds %>% dataloader(batch_size = 8)
dl$.length()

iter <- dl$.iter()
b <- iter$.next()
b

net <- nn_module(
  "CO2Net",
  initialize = function(num_countries,
                        num_regions,
                        country_emb_dim,
                        region_emb_dim,
                        num_numeric,
                        fc1_dim,
                        fc2_dim) {
    self$country_embedding <-
      nn_embedding(num_countries, country_emb_dim)
    self$region_embedding <-
      nn_embedding(num_regions, region_emb_dim)
    self$fc1 <- nn_linear(num_numeric, fc1_dim)
    self$fc2 <- nn_linear(fc1_dim + country_emb_dim + region_emb_dim, fc2_dim)
    self$output <- nn_linear(fc2_dim, 1)
  },
  forward = function(x) {
    x_country_emb <- self$country_embedding(x[ , 1, drop = FALSE]) %>%
      nnf_relu()
    x_region_emb <- self$region_embedding(x[ , 2, drop = FALSE]) %>%
      nnf_relu()
    x_numeric <- self$fc1(x[ , 3:-1]) %>%
      nnf_relu()
    self$fc2(torch_cat(x_country_emb, x_region_emb, x_numeric)) %>%
      nnf_relu() %>%
      self$output()
  }
)

num_countries <- as.factor(co2_data$country) %>% nlevels()
num_regions <- as.factor(co2_data$region) %>% nlevels()
country_emb_dim <- 7
region_emb_dim <- 4
num_numeric <- 8
fc1_dim <- 64
fc2_dim <- 64

model <- net(num_countries,
             num_regions,
             country_emb_dim,
             region_emb_dim,
             num_numeric,
             fc1_dim,
             fc2_dim)

optimizer <- optim_adam(model$parameters)

for (epoch in 1:10) {

  l <- c()

  for (b in enumerate(dl)) {
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- nnf_nll_loss(output, b[[2]])
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  }

  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}

