
# training a model using data augmentation
# based on Chapter 5.2.5 of Deep Learning with R


library(keras)
library(reticulate)

base_dir <- "~/Desktop/hotdog-data"
train_dir <- file.path(base_dir, "train")
val_dir <- file.path(base_dir, "validation")

# create data generators
# basically, these read in files from disk in batches and convert them to 
#  tensors

datagen <- image_data_generator(
  rescale = 1 / 255,
  rotation_range = 120,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  class_mode = "binary",
  batch_size = 20
)

val_generator <- flow_images_from_directory(
  val_dir,
  datagen,
  target_size = c(150, 150),
  class_mode = "binary",
  batch_size = 10
)

# 3 convolutional layers with dropout and early stopping
# 1,240,193 trainable parameters
model3conv <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

early_stop_list <- list(
  callback_early_stopping(monitor = "val_loss", patience = 2)
)

model3conv_hist <- model3conv %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# 4 convolutional layers with dropout and early stopping
# 1,846,977 trainable parameters
model4conv <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model4conv %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model4conv_hist <- model4conv %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# 4 convolutional layers with dropout and early stopping
#  1/4 the units in the dense layer at the end
# 642,369 trainable parameters
model4conv2 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model4conv2 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model4conv2_hist <- model4conv2 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )

# 5 convolutional layers with dropout and early stopping
#  1/4 the units in the dense layer at the end
#  421,313 trainable parameters
model5conv <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model5conv %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model5conv_hist <- model5conv %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )