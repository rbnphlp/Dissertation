
''' Creata a single layer NN using tensorflow , report its accuracy for a 1K iterations -- labelled as Model 1 in Report ----------------''

100 images at a time - batch - 10 digits : MNIST dataset '''


'Output : Accuracy  after 10K iteration:test or train  accuracy  :,60K images , batch size of 100, 600 iterations to com' \
'plete one epoch ,  '

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons

# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Test :10K images +labels , train :60 K images + labels

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)


## Place holders for the X and the Y_ labels:

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])

# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))  ## Variables are all the parameters that your training al
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The m00000odel
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
# GradienctDescent optimiser : optimising the weights and biases , where it is know to minimise the loss function :
#cross entropy .


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

for i in range(10000):
    #load batch of images and labells
    batch_X ,batch_Y=mnist.train.next_batch(100)
    train_data={X:batch_X,Y_:batch_Y}
    # train
    # Whats is the train step : It computes the gradient , dereives the deltas for the weights and biases  and updates the
    # weights and biases .: It makes the weights and biases move in the right direction
    sess.run(train_step,feed_dict=train_data)


    if (i%100==0):
        #check for every 100 iterations what the sucess  is on training and test it :
        #sucess?
        train_a,train_c=sess.run([accuracy,cross_entropy],feed_dict=train_data)
        print(str(i) + ": Resubstitution  accuracy:" + str(train_a) + " loss: " + str(train_c))

        # sucess on test data?
        test_data={X:mnist.test.images,Y_:mnist.test.labels}
        test_a,test_c=sess.run([accuracy,cross_entropy],feed_dict=test_data)
        print(str(i) + ": accuracy:" + str(test_a) + " loss: " + str(test_c))


print("train accuracy at " + str(i)+" th iteration : "+str(train_a*100)+"%")

print("accuracy using test datset at " + str(i)+" th iteration : "+str(test_a*100)+"%")


