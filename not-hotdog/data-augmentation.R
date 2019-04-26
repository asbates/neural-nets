
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
# highest val_acc: 62%
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
# highest val_acc: 51%
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
# highest val_acc: 52%
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
# highest val_acc: 51%
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


# save_model_hdf5(model3conv,
#                 "~/Desktop/hotdog_models/2019-04-26-3conv-data-aug.h5")
# save_model_hdf5(model4conv,
#                 "~/Desktop/hotdog_models/2019-04-26-4conv-data-aug.h5")
# save_model_hdf5(model4conv2,
#                 "~/Desktop/hotdog_models/2019-04-26-4conv2-data-aug.h5")
# save_model_hdf5(model4conv,
#                 "~/Desktop/hotdog_models/2019-04-26-5conv-data-aug.h5")

# plot(model3conv_hist)
# plot(model4conv_hist)
# plot(model4conv2_hist)
# plot(model5conv_hist)

# based on the plots it seems like less convolutional layers is better
# let's see if 2 convolutional layers will be better

# 2 convolutional layers with dropout and early stopping
# 5,327,937 trainable parameters
# highest val_acc: 57%
model2conv <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model2conv %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model2conv_hist <- model2conv %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# not as good as 3 layers
# let's try more units in the dense layer

# 3 convolutional layers with dropout and early stopping
# 128 units in last dense layer
# 2,424,065 trainable parameters
# highest val_acc: 61%
model3conv128 <- keras_model_sequential() %>% 
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


# 3 convolutional layers with dropout and early stopping
# 256 units in last dense layer
# 4,791,809 trainable parameters
# highest val_acc: 61%
model3conv256 <- keras_model_sequential() %>% 
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



# 3 convolutional layers with dropout and early stopping
# 2 dense layers
# 4,824,577 trainable parameters
# highest val_acc: 63%
model3conv2dense <- keras_model_sequential() %>% 
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
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv2dense %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model3conv2dense_hist <- model3conv2dense %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# 3 convolutional layers with dropout and early stopping
# 2 dense layers with elastic net regularization
# 4,824,577 trainable parameters
# highest val_acc: 50%
model3conv2densereg <- keras_model_sequential() %>% 
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
  layer_dense(units = 256, 
              activation = "relu",
              kernel_regularizer = regularizer_l1_l2(0.001, 0.001)) %>% 
  layer_dense(units = 128,
              activation = "relu",
              kernel_regularizer = regularizer_l1_l2(0.001, 0.001)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv2densereg %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model3conv2densereg_hist <- model3conv2densereg %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# maybe i should have been monitoring validation accuracy instead of loss
early_stop_list <- list(
  callback_early_stopping(monitor = "val_acc", patience = 2)
)


# 2 convolutional layers with dropout and early stopping
# 2 dense layers with elastic net regularization
#  trainable parameters
# highest val_acc: 56%
model2conv2densereg <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, 
              activation = "relu",
              kernel_regularizer = regularizer_l1_l2(0.001, 0.001)) %>% 
  layer_dense(units = 128,
              activation = "relu",
              kernel_regularizer = regularizer_l1_l2(0.001, 0.001)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model2conv2densereg %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model2conv2densereg_hist <- model2conv2densereg %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# 3 convolutional layers with dropout and early stopping
# more filters in 3rd layer
# 2 dense layers
#  trainable parameters
# highest val_acc: 67%
model3conv2dense2 <- keras_model_sequential() %>% 
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
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, 
              activation = "relu") %>% 
  layer_dense(units = 128,
              activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv2dense2 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model3conv2dense2_hist <- model3conv2dense2 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# what about 1 conv layer and more dense layers?
# should have tried something simpler first!
model1conv3dense <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, 
              activation = "relu") %>% 
  layer_dense(units = 128,
              activation = "relu") %>% 
  layer_dense(units = 128,
              activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model1conv3dense %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model1conv3dense_hist <- model1conv3dense %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )

# nope


# seems like more filters is the way to go
# highest val_acc: 54%
model3conv2dense3 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, 
              activation = "relu") %>% 
  layer_dense(units = 128,
              activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv2dense3 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model3conv2dense3_hist <- model3conv2dense3 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )

# highest val_acc: 65%
model3conv2dense4 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128,
                kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, 
              activation = "relu") %>% 
  layer_dense(units = 128,
              activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model3conv2dense4 %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("acc")
  )

model3conv2dense4_hist <- model3conv2dense4 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = 10,
    callbacks = early_stop_list
  )


# ok, it seems like 
# 3 conv layers with 32, 64, 128 layers
# 2 dense layers is the best i'm going to get

plot(model3conv2dense2_hist)

# save_model_hdf5(model3conv2dense2,
#                 "~/Desktop/hotdog_models/2019-04-26-3conv2dense-data-aug.h5")

# let's see how it does on the test set
test_dir <- file.path(base_dir, "test")
test_datagen <- image_data_generator(rescale = 1/255)
test_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  class_mode = "binary",
  batch_size = 20
)

# wait! i just realized i was augmenting the validation data!

