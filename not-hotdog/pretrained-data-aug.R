

library(keras)
library(reticulate)

base_dir <- "~/Desktop/hotdog-data"
train_dir <- file.path(base_dir, "train")
val_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

train_datagen <- image_data_generator(
  rescale = 1 / 255,
  rotation_range = 120,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE)

test_datagen <- image_data_generator(rescale = 1 / 255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  class_mode = "binary",
  batch_size = 20
)

val_generator <- flow_images_from_directory(
  val_dir,
  test_datagen,
  target_size = c(150, 150),
  class_mode = "binary",
  batch_size = 10
)


# start with a conv base from vgg16 as in Chapter 5.3.1
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

freeze_weights(conv_base)

# 256 units in dense layer
# val_acc: 90%
# wow! that's a huge difference!
one_dense256 <- keras_model_sequential() %>% 
  conv_base() %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

one_dense256 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

one_dense256_hist <- one_dense256 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10
  )



# 128 units in dense layer
# val_acc: 87%
one_dense128 <- keras_model_sequential() %>% 
  conv_base() %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

one_dense128 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

one_dense128_hist <- one_dense128 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10
  )



# 256 units in dense layer with dropout
# train longer
# val_acc: 84%
one_dense256drop <- keras_model_sequential() %>% 
  conv_base() %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

one_dense256drop %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

one_dense256drop_hist <- one_dense256drop %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 50,
    validation_data = val_generator,
    validation_steps = 10
  )



plot(one_dense256_hist)
plot(one_dense128_hist)
plot(one_dense256drop_hist)


save_model_hdf5(one_dense256,
                "~/Desktop/hotdog_models/2019-04-26-vgg16-256.h5")

save_model_hdf5(one_dense256,
                "~/Desktop/hotdog_models/2019-04-26-vgg16-128.h5")

save_model_hdf5(one_dense256,
                "~/Desktop/hotdog_models/2019-04-26-vgg16-256drop.h5")















