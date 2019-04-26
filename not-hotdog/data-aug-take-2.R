
# it seems like using keras default image size of 256 x 256 is helping
# maybe this combined with data augmentation will be better?

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
  target_size = c(256, 256),
  class_mode = "binary",
  batch_size = 20
)

val_generator <- flow_images_from_directory(
  val_dir,
  test_datagen,
  target_size = c(256, 256),
  class_mode = "binary",
  batch_size = 10
)


early_stop_list <- list(
  callback_early_stopping(monitor = "val_acc", patience = 2)
)


# ============================
# ==== 3 conv layers ========
# ===========================

# 256 units in dense layer
# 14,802,433 trainable parameters
# val_acc: 60%
model3conv256 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
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
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv256 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model3conv256_hist <- model3conv256 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# 128 units in dense layer
# 7,429,377  trainable parameters
# val_acc: 60%
model3conv128 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
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
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv128 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model3conv128_hist <- model3conv128 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# 128 units in dense layer
# 128 filters in last conv layer
#   trainable parameters
# val_acc: 62%
model3conv128128 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv128128 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model3conv128128_hist <- model3conv128128 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# 128 units in dense layer
# 128 filters in last conv layer
# dropout after every layer
# 14,839,105  trainable parameters
# val_acc: 57%
model3conv128128drop <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv128128drop %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model3conv128128drop_hist <- model3conv128128drop %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )



# ============================
# ==== 2 conv layers ========
# ===========================

# 256 units in dense layer
# 63,000,001 trainable parameters
# val_acc: 60%
model2conv256 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model2conv256 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model2conv256_hist <- model2conv256 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# 128 units in dense layer
# 31,509,697 trainable parameters
# val_acc: 61%
model2conv128 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model2conv128 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model2conv128_hist <- model2conv128 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )



# 64 units in dense layer
# val_acc: 60%
model2conv64 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model2conv64 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model2conv64_hist <- model2conv64 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )




# =========================
# ===== 4 conv layers =====
# =========================

# different optimizer as well
# and more dropout
# val_acc: 52%
model4conv128adam <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model4conv128adam %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr = 1e-5),
    metrics = c("acc")
  )

model4conv128adam_hist <- model4conv128adam %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )

# am i stopping too early?

# val_acc: 63%
model3conv128128ns <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv128128ns %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model3conv128128ns_hist <- model3conv128128ns %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 20,
    validation_data = val_generator,
    validation_steps = 10
  )

