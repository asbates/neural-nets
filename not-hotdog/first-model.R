
# this script builds a simple model to be used as a prototype in order to get a
#  working Shiny app going
# see Deep Learning with R chapter 5.2.3/4 which is where this code was derived

library(keras)

base_dir <- "~/Desktop/hotdog-data"
train_dir <- file.path(base_dir, "train")
val_dir <- file.path(base_dir, "validation")

# create data generators
# basically, these read in files from disk in batches and convert them to 
#  tensors

train_datagen <- image_data_generator(rescale = 1 / 255)
val_datagen <- image_data_generator(rescale = 1 / 255)

# notes: 
# the images are 512 x 512
# the default size in flow_images_from_directory() is 256 x 256
# classes will automatically be inferred based on subdirectory names
#  and labels created
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  class_mode = "binary",
  batch_size = 20
)

val_generator <- flow_images_from_directory(
  val_dir,
  val_datagen,
  class_mode = "binary",
  batch_size = 10
)

# example from text to see what generators do
# batch <- generator_next(train_generator)
# str(batch)


model <- keras_model_sequential() %>% 
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
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model_hist <- model %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 40,  # 40 steps x 20 batches each step = 800 images
    epochs = 10,
    validation_data = val_generator,
    validation_steps = 10
  )

# first epoch took about 10 seconds, subsequen took about 6
# that's a lot faster than i originally thought
# validation accuracy up to about 67%

# since this didn't take too long let's go ahead and add a couple layers
# let's also resize the images before feeding to model to reduce total
#  number of parameters

train_generator_3conv <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  class_mode = "binary",
  batch_size = 20
)

val_generator_3conv <- flow_images_from_directory(
  val_dir,
  val_datagen,
  target_size = c(150, 150),
  class_mode = "binary",
  batch_size = 10
)

model_3conv <- keras_model_sequential() %>% 
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
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model_3conv)

model_3conv %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )

model_3conv_hist <- model_3conv %>% 
  fit_generator(
    train_generator_3conv,
    steps_per_epoch = 40,  # 40 steps x 20 batches each step = 800 images
    epochs = 30,
    validation_data = val_generator_3conv,
    validation_steps = 10
  )

# about 5-7 seconds per epoch
# validation accuracy not really much better

plot(model_hist)
plot(model_3conv_hist)

# for some reason this isn't working
# h5py is installed but it's prbably something about the python path
# model %>% save_model_hdf5("not-hotdog/models/2019-04-22-2conv.h5")

# come back later and figure out how to do this

