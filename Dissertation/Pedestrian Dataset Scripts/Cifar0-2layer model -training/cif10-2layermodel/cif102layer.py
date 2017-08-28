# Using a pretrained model using Cifar10 dataset
import tensorflow as tf
    # .Optimizer

'''Network archicture in accodance with : https://github.com/tflearn/tflearn/blob/master/examples/basics/finetuning.py'''

from labelimagesconfustionmatrix import * 
# Data loading

num_classes = 2

# Redefinition of convnet_cifar10 network
network = input_data(shape=[None, IMG_SIZE,IMG_SIZE, 1],name='input') # changed from 3 to 2
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.75)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
#Finetuning Softmax layer (Setting restore=False to not restore its weights)
softmax = fully_connected(network, num_classes, activation='softmax')
opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
regression = regression(softmax, optimizer=opt,
                        loss='categorical_crossentropy',
                        # learning_rate=LR,
                        name="targets")


model = tflearn.DNN(regression, tensorboard_dir='log')

