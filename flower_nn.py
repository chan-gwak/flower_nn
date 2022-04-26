# flower_nn.py
# by Chan Gwak

# A script that trains and tests a neural network to categorize
# images of flowers into one of seventeen categories.
# Uses a modified version of 17 Category Flower Dataset
# (www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).

# "50x50flowers.images.npy" contains 1360 images,
# each 50x50 pixels, each pixel having 3 values.
# (Checking gives a range of 0~255, i.e. RGB.)
# If loaded as imgs, then imgs[0][25][25][0] contains
# a value like 181.0, which is of the numpy.float64 class.

# "50x50flowers.targets.npy" contains 1360 float64's,
# each of which is a number in [1.0,2.0,...,17.0].
# (Checking reveals that there are 80 of each. 17*80=1360)

# Best parameters thus far:
# 3 convolutional layers
# 120 dense hidden layer neurons
# Adam with learning rate 1e-3 (default)
# Batch size 128
# 1500 Epochs (images are augmented every epoch)
# Dropout after third pooling layer w/ rate 0.5
# Dropout after dense hidden layer w/ rate 0.5
#
# This should give a training accuracy of about 93~94%
# and a testing accuracy of 74~82%.


import numpy as np
import tensorflow.keras.utils as ku
import tensorflow.keras.preprocessing.image as kpi
import sklearn.model_selection as skms
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
from matplotlib import pyplot as plt


# Hyperparameters
train_set_size = 0.8 # train_size; Use 0.2 of data for testing
num_ftmap_1 = 50 # numfm_1; number of feature maps in 1st convolutional layer
num_ftmap_2 = 50 # numfm_2; number of feature maps in 2nd convolutional layer
num_ftmap_3 = 100 # numfm_3; number of feature maps in 3rd convolutional layer
conv_sq_1 = (7, 7) # kernel_size; size of sampling square in 1st convolutional layer
pool_sq_1 = (2, 2) # pool_size; size of square in 1st max-pooling layer
pool_str_1 = (2, 2) # strides; translation amount for 1st pooling, default (1, 1)
conv_sq_2 = (3, 3) # kernel_size for 2nd conv. layer
pool_sq_2 = (2, 2) # pool_size for 2nd pooling layer
pool_str_2 = (2, 2) # strides for 2nd pooling layer
conv_sq_3 = (3, 3) # kernel_size for 3rd conv. layer
pool_sq_3 = (2, 2) # pool_size for 3rd pooling layer
pool_str_3 = (2, 2) # strides for 3rd pooling layer
drop_rate_1 = 0.5 # rate of dropout after 3rd pooling layer
drop_rate_2 = 0.5 # rate of dropout after dense hidden layer
num_hidden = 120 # numnodes; number of neurons in dense hidden layer
opt_learn_rate = 1e-3 # learning_rate of the optimizer; 1e-3 is default for Adam
opt_batch_size = 128 # batch_size for sgd & variants; default 32
num_epochs = 1500 # number of epochs for training

# Augmentation parameters
rota_intv=45 # default 0, max. rotation angle in degrees
xtrans_intv=0.2 # default 0.0, float means % to translate by
ytrans_intv=0.2 # default 0.0, float means % to translate by
bright_intv=(0.8,1.2) # default None, % for (min,max) brightness
shear_intv=0.2 # default 0.0, shearing angle in degrees
zoom_intv=0.2 # default 0.0, % to zoom in or out by (at most)
fill_type="nearest" # default "nearest", fill empty areas w/ nearest colors
horiz_flip=True # default False, do a horizontal flip
verti_flip=True # default False, do a vertical flip
pixel_rescale=1/255 # rescale; divide RGB values by 255.

# Settings
verbosity = 1 # decides the verbosity of model.fit

# Names of files dataset and associated targets.
# These should be colocated with this script.
img_fname = '50x50flowers.images.npy'
tgt_fname = '50x50flowers.targets.npy'


# Load the files.
print("\nReading flowers input file.")
img = np.load(img_fname)
print("Reading flowers target file.")
tgt = np.load(tgt_fname)

# Get the shape of the data.
data_size, img_x, img_y, nchan = np.shape(img)
data_size_check = np.size(tgt)

# Minor check: Make sure the input and target files have the 
# same number of data points (#images) as they should match up.
if data_size == data_size_check:
	print("Good: The input and target files both have "+
		str(data_size)+" data points.")
else:
	print("Bad: The input has data size "+str(data_size)+
		" but the target has "+str(data_size_check)+".")
	raise SystemExit(0)

# Get the range of category (or class) labels.
min_ctg = np.min(tgt) # First label
max_ctg = np.max(tgt) # Last label
num_ctg = int(max_ctg - min_ctg + 1) # Range of class labels, 17 expected

# Change the target labels to categorical format (one-hot encoding).
# Note: to_categorial takes a class vector of integers from 0 to num_classes,
#       so subtract ones to set the first label to zero.
tgt = ku.to_categorical(tgt - min_ctg*np.ones(data_size), num_ctg)

# Split the data into training and testing sets. (Shuffled by default pre-split.)
train_img, test_img, train_tgt, test_tgt = skms.train_test_split(img, tgt, 
	train_size = train_set_size)


# A function to build the model.
# As this is an image-processing problem use convolutional layers,
# each followed by batch normalization and a max-pooling layer.
# The 1st conv. layer has [numfm_1] feature maps which takes an image's 
# [img_x*img_y*nchan] values (50x50 pixels, x3 for RGB).
# The model ends with a fully-connected hidden layer of [numnodes] neurons,
# and an output layer that outputs one of [num_ctg] categories.
# Dropouts occur after the 3rd pooling layer and the dense hidden layer,
# to reduce overfitting.
def build_model(numfm_1, numfm_2, numfm_3, numnodes):

	model = km.Sequential()

	model.add(kl.Conv2D(numfm_1, kernel_size = conv_sq_1, 
		input_shape = (img_x, img_y, nchan), 
		name = 'convolutional_1', activation = 'relu'))
	model.add(kl.BatchNormalization(name = 'batchnorm_1'))
	model.add(kl.MaxPooling2D(pool_size = pool_sq_1, strides = pool_str_1,
		name = 'pooling_1'))

	model.add(kl.Conv2D(numfm_2, kernel_size = conv_sq_2,
		name = 'convolutional_2', activation = 'relu'))
	model.add(kl.BatchNormalization(name = 'batchnorm_2'))
	model.add(kl.MaxPooling2D(pool_size = pool_sq_2, strides = pool_str_2,
		name = 'pooling_2'))

	model.add(kl.Conv2D(numfm_3, kernel_size = conv_sq_3,
		name = 'convolutional_3', activation = 'relu'))
	model.add(kl.BatchNormalization(name = 'batchnorm_3'))
	model.add(kl.MaxPooling2D(pool_size = pool_sq_3, strides = pool_str_3,
		name = 'pooling_3'))

	model.add(kl.Dropout(drop_rate_1))

	model.add(kl.Flatten())
	model.add(kl.Dense(numnodes, name = 'dense_hidden', activation = 'tanh'))

	model.add(kl.Dropout(drop_rate_2))

	# As this is a categorization problem, the output layer should use softmax.
	model.add(kl.Dense(num_ctg, name = 'output', 
		activation = 'softmax'
	))

	return model


# Build the layer with [num_hidden] neurons in the hidden layer.
print("Building network.\n")
model = build_model(num_ftmap_1, num_ftmap_2, num_ftmap_3, num_hidden)

# Compile the model with Adam (a variant of sgd) as the optimization algorithm, 
# and categorical cross-entropy as the cost function.
opt = ko.Adam(learning_rate=opt_learn_rate) # default is 1e-3 for Adam
print("\nCompiling the model.")
model.compile(optimizer = opt, metrics = ['accuracy'], loss = 'categorical_crossentropy')

print("Printing a summary of the model:\n")
model.summary()


# To increase the amount of data, generate more images via translations, 
# rotations, reflections, shearing, zooming, and brightness changes.
# Also rescale the RGB values to fit in [0,1], for better learning.
# This generator will be used in model-fitting later.
datagen = kpi.ImageDataGenerator(
    rotation_range=rota_intv,
    width_shift_range=xtrans_intv,
    height_shift_range=ytrans_intv,
    brightness_range=bright_intv,
    shear_range=shear_intv,
    zoom_range=zoom_intv,
    fill_mode=fill_type,
    horizontal_flip=horiz_flip,
    vertical_flip=verti_flip,
    rescale=pixel_rescale,
)

# Fit the model on the training data, augmenting in real time.
# The network will see a new set of augmented data every epoch.
print("\nTraining network with real-time augmentation.\n")
fit = model.fit(
	datagen.flow(train_img, train_tgt, batch_size = opt_batch_size),
	# Specify steps_per_epoch for augmented sets
	steps_per_epoch = (train_set_size * data_size) // opt_batch_size,
	epochs = num_epochs, verbose = verbosity)

# Evaluate the model using the testing data.
print("\nEvaluating the model on testing data, rescaled like the training data.")
score = model.evaluate(test_img * pixel_rescale, test_tgt)

print("\nThe testing score is:")
print(score)

print("\nHere is a summary of the model again:\n")
model.summary()


# Now plot the loss over time.
print("\nPlotting loss.")
plt.plot(fit.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('loss.png')
