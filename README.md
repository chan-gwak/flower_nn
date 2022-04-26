# flower_nn.py
# by Chan Gwak

A script that trains and tests a neural network to categorize images of flowers into one of seventeen categories.
Uses a modified version of 17 Category Flower Dataset (www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).

"50x50flowers.images.npy" contains 1360 images, each 50x50 pixels, each pixel having 3 values (a range of 0~255, i.e. RGB).
If loaded as imgs, then imgs[0][25][25][0] contains a value like 181.0, which is of the numpy.float64 class.

"50x50flowers.targets.npy" contains 1360 float64's, each of which is a number in [1.0,2.0,...,17.0].
(There are 80 of each. 17*80=1360)

Best parameters thus far: 
* 3 convolutional layers
* 120 dense hidden layer neurons
* Adam with learning rate 1e-3 (default)
* Batch size 128
* 1500 Epochs (images are augmented every epoch)
* Dropout after third pooling layer w/ rate 0.5
* Dropout after dense hidden layer w/ rate 0.5

This should give a training accuracy of about 93-94% and a testing accuracy of 74-82%.
