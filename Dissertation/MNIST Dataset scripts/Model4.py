''' Implement the Covnet model for the MNIST dataset- Model 4 convolutional neural network  '''



## THe Relu function : simpler and performs better than the sigmoid function for deep NN
# Works similar to biological neurons : 0 when no signal ..
## Difference between sgmoid functions : change the biases to  small positive values :




'100 images at a time - batch - 10 digits : MNIST dataset '''


'Output : Accuracy  after 200 0 iterations: '
"Accuracy for training data set : 100% !!!"
"accuracy for test dataset : 97.32%"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

#
# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)


##Initialise some variables to store for plotting
trainaccuracy=np.array([])
testaccuracy=np.array([])

cross_entropy_train=np.array([])
cross_entroppy_test=np.array([])


## Place holders for the X and the Y_ labels:

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])

# Placeholder for the adamoptimizer (define the placeholder and use it for adamoptimizer and then later populate with
# the results from the run session
lr=tf.placeholder(tf.float32)




## Initialise the output  channels or feature maps  (no more neurons as the same as pixels)

L=4         # first convolutional layer
M=8         # 2nd convolutional layer
N=12        # 3rd convolutional layer
O=200       # Fully connected layers
P=10        # Final Output layer

# Input :
# An image of 28*28*1 instead of  a single vector of 784*1

# 5 weights to be initiallised using truncated normal  :
#shape[5,5,1,4] : corresponds to : 5,5 patch size for the scanning , 1 input channel , 4 output channel
W1=tf.Variable(tf.truncated_normal([5,5,1,L],stddev=0.1)) # 28*28*4 1st CN # Filter size (

W2=tf.Variable(tf.truncated_normal([4,4,L,M],stddev=0.1)) #14*14*8 as 8  output  , notice stride =2 as it halves


W3=tf.Variable(tf.truncated_normal([4,4,M,N],stddev=0.1)) #7*7*12 stride:2

W4=tf.Variable(tf.truncated_normal([7*7*12,O],stddev=0.1))# Fully connected layer: Input 7*7*12 ,output:200 Neurons

W5=tf.Variable(tf.truncated_normal([O,P],stddev=0.1)) # Input :200 Neuron , Output :10 classes


#Initialise the Biases
b1 = tf.Variable(tf.ones([L])/10)
b2 = tf.Variable(tf.ones([M])/10)
b3 = tf.Variable(tf.ones([N])/10)
b4 = tf.Variable(tf.ones([O])/10)
b5 = tf.Variable(tf.ones([P])/10)

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
#XX = tf.reshape(X, [-1, 784]) -- No more flattening of the input image required


## Build the model :

# stride notation [1,H,W,1] : H is the height ,W is the width as the neuron is scanning in both directions
# First layer :
stride=1
Y1cn=tf.nn.conv2d(X,W1,strides=[1,stride,stride,1],padding='SAME') # padding :output map size as the same as input map size
# stride defines the amount of overlapness between each convoluion window : A stride of 5 , for a patch size of 5 weill#
# mean no overlapping convolution windows when scanning .

Y1=tf.nn.relu(Y1cn+b1)


#2nd Layer :
stride=2
Y2cn=tf.nn.conv2d(Y1,W2,strides=[1,stride,stride,1],padding='SAME') # padding :output map size as the same as input map size
# stride defines the amount of overlapness between each convoluion window : A stride of 5 , for a patch size of 5 weill#
# mean no overlapping convolution windows when scanning .

Y2=tf.nn.relu(Y2cn+b2)

#3rd Layer :
stride=2
Y3cn=tf.nn.conv2d(Y2,W3,strides=[1,stride,stride,1],padding='SAME') # padding :output map size as the same as input map size
# stride defines the amount of overlapness between each convoluion window : A stride of 5 , for a patch size of 5 weill#
# mean no overlapping convolution windows when scanning .

Y3=tf.nn.relu(Y3cn+b3)
#4th Layer(full CN layer) :
# Use Reshape to convert from a convolution layer to fully connected layer :
YY=tf.reshape(Y3,shape=[-1,7*7*N])


Y4=tf.nn.relu(tf.matmul(YY,W4)+b4)
# stride defines the amount of overlapness between each convoluion window : A stride of 5 , for a patch size of 5 weill#
# mean no overlapping convolution windows when scanning .

# Final Out put function from 200 Neurons :
## Replace Y4 with a logits function :
Ylogits=tf.matmul(Y4,W5)+b5
Y=tf.nn.softmax(Ylogits)






# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch

cross_entropy_logit=tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy_logit) * 100.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005

# Adam optimizer : Saddle points frequent in 10K weights and biases .
#Points are not local minima , but where the gradient is neverthless and get stuck there .
# Adam optimizer gets around that problem :
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy) ## Lr added

## Define  the session , so the nodes can be executed :
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)



'''Explanation of the below loop :'''
## Feed dictionary , feed on all the missing data that the placeholders : get images 100 images at a time
# Training loop :pass 100 images at a time , train it ;
# COmpute the gradient on the current 100 images and labels , add alittle fraction of this to our cweights and biases :train step
# Start over with a new batch of images and labels :(60K) :100K if 1000 iterations :
# Learning rate : little fraction of gradient that you will be adding to the weights and baises: how to choose the right gradient ?


#Tensor flow has a deffered excecution model :the tf. statements only produces a computation graph
# Tf done primaarly done for distributed computing ,helps a lot with distributing the graphs to multiple systems :
# sess.run excutes the nodes : i.e need to run this everytime you need to execute something + a feed dict _


''' Learning rate decay added to the relu code :'''
#Define a min learning and max min rate :
lrmin=0.0001
lrmax=0.003



for i in range(10000+1):
    '''Optimizing using  decying learning rate:'''
    # feed the placeholder with the lrdecay with each iteration:
    learningrate = lrmin + (lrmax - lrmin) * np.exp(-i / 2000)




    #load batch of images and labells
    batch_X ,batch_Y=mnist.train.next_batch(100)

    # Add in the learning rate :
    train_data={X:batch_X,Y_:batch_Y,lr:learningrate}

    # Whats is the train step : It computes the gradient , dereives the deltas for the weights and biases  and updates the
    # weights and biases .: It makes the weights and biases move in the right direction
    sess.run(train_step,feed_dict=train_data)



    if (i%100==0):
        #check for every 100 iterations what the sucess  is on training and test it :
        #sucess?
        train_a,train_c=sess.run([accuracy,cross_entropy],feed_dict={X:batch_X,Y_:batch_Y})
        #print(str(i) + ": Resubstitution  accuracy:" + str(train_a) + " loss: " + str(train_c))

        # sucess on test data?
        test_data={X:mnist.test.images,Y_:mnist.test.labels}
        test_a,test_c=sess.run([accuracy,cross_entropy],feed_dict=test_data)
        #print(str(i) + ": accuracy:" + str(test_a) + " loss: " + str(test_c))

        #populate the train and test accuract for visualising the plotting :
        trainaccuracy=np.append(trainaccuracy,train_a)
        testaccuracy=np.append(testaccuracy,test_a)
        #populate the cross entropy for train and test data :

        cross_entropy_train=np.append(cross_entropy_train,train_c)
        cross_entroppy_test=np.append(cross_entroppy_test,test_c)


print("train accuracy at " + str(i)+" th iteration : "+str(train_a*100)+"%")
print("accuracy using test datset at " + str(i)+" th iteration : "+str(test_a*100)+"%")



## Function for plotting train and test  accuracy for 2K iterations :
# use matplot from the pyplot library to plot the  functions :
import matplotlib.pyplot as plt

#

#plt.subplot(211)
#create legends.
x1=list(range(len(trainaccuracy)))
y1=trainaccuracy
x2=list(range(len(testaccuracy)))
y2=testaccuracy



plt.plot(x1,y1,c='r',label='Training accuracy')
plt.plot(x2,y2,c='g',label='Test accuracy')
plt.legend()
plt.show()


## What happens to the cross entropy loss :
x3=list(range(len(cross_entropy_train)))
y3=cross_entropy_train
x4=list(range(len(cross_entroppy_test)))
y4=cross_entroppy_test



plt.plot(x3,y3,c='r',label='training -cross entropy loss')
plt.plot(x4,y4,c='g',label='test -cross entropy loss')
plt.legend()
plt.show()

