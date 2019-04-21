
# this file alters the original train/test format
# the original setup is 249 training images and 250 test images
# actually, 249/250 each of hot dog and not hot dog
# we will instead use 400 for training, 50 for validation, and 49 for testing
# the dataset isn't very large so I want to have as much data as possible for 
#  training

# install.packages("magick")
library(magick)

# =====================================
# ======= read in original files ======
# =====================================

orig_train_hotdog_files <- list.files("~/Desktop/seefood/train/hot_dog")
orig_train_nothotdog_files <- list.files("~/Desktop/seefood/train/not_hot_dog")
orig_test_hotdog_files <- list.files("~/Desktop/seefood/test/hot_dog")
orig_test_nothotdog_files <- list.files("~/Desktop/seefood/test/not_hot_dog")

hotdog_files <- c(orig_train_hotdog_files, orig_test_hotdog_files)
not_hotdog_files <- c(orig_train_nothotdog_files, orig_test_nothotdog_files)

orig_train_hotdog_path <- "~/Desktop/seefood/train/hot_dog"
orig_test_hotdog_path <- "~/Desktop/seefood/test/hot_dog"
orig_train_nothotdog_path <- "~/Desktop/seefood/train/not_hot_dog"
orig_test_nothotdog_path <- "~/Desktop/seefood/test/not_hot_dog"

num_train_hotdog <- length(orig_train_hotdog_files)
num_test_hotdog <- length(orig_test_hotdog_files)
num_train_not_hotdog <- length(orig_train_nothotdog_files)
num_test_not_hotdog <- length(orig_test_nothotdog_files)
num_hotdog <- num_train_hotdog + num_test_hotdog
num_not_hotdog <- num_train_not_hotdog + num_test_not_hotdog


hotdog_images <- vector("list", num_hotdog)
not_hotdog_images <- vector("list", num_not_hotdog)

for (i in 1:num_train_hotdog) {
  hotdog_images[[i]] <- 
    image_read(
      paste0(orig_train_hotdog_path,
             "/",
             orig_train_hotdog_files[[i]])
      )
}

for (i in (num_train_hotdog + 1):num_hotdog) {
  hotdog_images[[i]] <- 
    image_read(
      paste0(orig_test_hotdog_path,
             "/",
             orig_test_hotdog_files[[i - num_train_hotdog]])
      )
}

names(hotdog_images) <- hotdog_files

for (i in 1:num_train_not_hotdog) {
  not_hotdog_images[[i]] <- image_read(
    paste0(orig_train_nothotdog_path,
           "/",
           orig_train_nothotdog_files[[i]])
    )
}

for (i in (num_train_not_hotdog +1): num_not_hotdog) {
  not_hotdog_images[[i]] <- image_read(
    paste0(orig_test_nothotdog_path,
           "/",
           orig_test_nothotdog_files[[i - num_train_not_hotdog]])
  )
}

names(not_hotdog_images) <- not_hotdog_files


# =========================================
# ====== write out with new structure =====
# =========================================

# shuffle
set.seed(817438)
hotdog_files <- sample(hotdog_files, num_hotdog)
not_hotdog_files <- sample(not_hotdog_files, num_not_hotdog)

n_train <- 400
n_val <- 50
n_test <- 49

train_hotdog_files <- hotdog_files[1:400]
val_hotdog_files <- hotdog_files[401:450]
test_hotdog_files <- hotdog_files[451:499]

train_nothotdog_files <- not_hotdog_files[1:400]
val_nothotdog_files <- not_hotdog_files[401:450]
test_nothotdog_files <- not_hotdog_files[451:499]

train_hotdog_images <- hotdog_images[train_hotdog_files]
val_hotdog_images <- hotdog_images[val_hotdog_files]
test_hotdog_images <- hotdog_images[test_hotdog_files]

train_nothotdog_images <- not_hotdog_images[train_nothotdog_files]
val_nothotdog_images <- not_hotdog_images[val_nothotdog_files]
test_nothotdog_images <- not_hotdog_images[test_nothotdog_files]

train_hotdog_path <- "~/Desktop/hotdog-data/train/hot_dog"
val_hotdog_path <- "~/Desktop/hotdog-data/validation/hot_dog"
test_hotdog_path <- "~/Desktop/hotdog-data/test/hot_dog"

train_nothotdog_path <- "~/Desktop/hotdog-data/train/not_hot_dog"
val_nothotdog_path <- "~/Desktop/hotdog-data/validation/not_hot_dog"
test_nothotdog_path <- "~/Desktop/hotdog-data/test/not_hot_dog"

# hot dog images
for (i in seq_along(train_hotdog_images)){
  image_write(
    train_hotdog_images[[i]],
    path = paste0(train_hotdog_path, 
                  "/",
                  names(train_hotdog_images)[[i]]),
    format = "jpeg"
  )
}

for (i in seq_along(val_hotdog_images)){
  image_write(
    val_hotdog_images[[i]],
    path = paste0(val_hotdog_path, 
                  "/",
                  names(val_hotdog_images)[[i]]),
    format = "jpeg"
  )
}

for (i in seq_along(test_hotdog_images)){
  image_write(
    test_hotdog_images[[i]],
    path = paste0(test_hotdog_path, 
                  "/",
                  names(test_hotdog_images)[[i]]),
    format = "jpeg"
  )
}


# not hot dog images
for (i in seq_along(train_nothotdog_images)){
  image_write(
    train_nothotdog_images[[i]],
    path = paste0(train_nothotdog_path, 
                  "/",
                  names(train_nothotdog_images)[[i]]),
    format = "jpeg"
  )
}

for (i in seq_along(val_nothotdog_images)){
  image_write(
    val_nothotdog_images[[i]],
    path = paste0(val_nothotdog_path, 
                  "/",
                  names(val_nothotdog_images)[[i]]),
    format = "jpeg"
  )
}

for (i in seq_along(test_nothotdog_images)){
  image_write(
    test_nothotdog_images[[i]],
    path = paste0(test_nothotdog_path, 
                  "/",
                  names(test_nothotdog_images)[[i]]),
    format = "jpeg"
  )
}
