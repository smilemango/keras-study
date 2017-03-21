from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

s = Sequential()

cifar10.load_data()