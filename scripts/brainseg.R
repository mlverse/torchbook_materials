library(torch)
library(torchvision)
library(dplyr)
library(magick)


train_dir <- "data/kaggle_3m_train"
valid_dir <- "data/kaggle_3m_valid"

brainseg_dataset <- dataset(
  name = "brainseg_dataset",

  initialize = function(images_dir,
                        transform = NULL,
                        image_size = 256,
                        random_sampling = TRUE
                        ) {

    volumes <- list()
    masks <- list()

    print("reading images...")
    dirs <- list.files(images_dir)
    for (d in dirs) {
      image_slices <- list()
      mask_slices <- list()
      patient_data <- list.files(file.path(images_dir, d))
      for (p in patient_data) {
        patient_id <- d
        img <- image_read(file.path(images_dir, d, p))
        if (grepl("mask", p)) {
         mask_slices <- c(mask_slices, img)
        } else {
          image_slices <- c(image_slices, img)
        }
      }
      volumes[d] <<- image_slices[2:(length(image_slices) - 1)]
      masks[d] <<- mask_slices[2:(length(mask_slices) - 1)]
    }


 to_tensor <- function(img) {
   img <- as.integer(magick::image_data(img, channels = "rgb"))
   img <- torch::torch_tensor(img)$permute(c(3,1,2))
   img <- img$to(dtype = torch::torch_float32())
   img <- img$contiguous()
   img <- img$div(255)
 }


    if len(image_slices) > 0:
      patient_id = dirpath.split("/")[-1]
    volumes[patient_id] = np.array(image_slices[1:-1])
    masks[patient_id] = np.array(mask_slices[1:-1])
    self.patients = sorted(volumes)
  },

  .getitem = function(index) {
    x <- self$data[index, 2:-1]
    y <- self$data[index, 1] - 1

    list(x, y)
  },

  .length = function() {
    self$data$size()[[1]]
  }

)
