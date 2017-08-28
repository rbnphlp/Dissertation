''' Plot the roc curves  and confusion matrixes for models '''

from __future__ import print_function

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve ,auc
#from confusionmatrixmodels  import *
import numpy as np 

##File to obtain confusion and roc curves for the best and worst performing models##########################################


from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
import sklearn.model_selection as sk
from sklearn.metrics import *



import numpy as np
from keras.models import load_model
# Save the model as png file
from keras.utils import plot_model
#from labelimagesconfustionmatrix import * 
import matplotlib.pyplot as plt

model_1 = load_model('bestmodel.h5') ### model with the best results 
model_2=load_model("badmodel.h5") ## model with the worst results 
#model_3=load_model("bestmodel.h5")
 # load the saved model
#train_data="/home/ha46/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/traindata.npy"
#test_data="/home/ha46/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/testdata.npy"

train_data="C:/Users/Creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/traindata.npy"
test_data="C:/Users/Creem//Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/testdata.npy"


IMG_SIZE=50
train_data=np.load(train_data)
test_data=np.load(test_data)	
	
#MODEL_NAME="ped-class--9-Preprocessed- 2layer cifar structure -Dropout deactivated-.model" ## Best Model-Predictions
#odel.save(MODEL_NAME)


#plot_model(model_1, to_file='model.png', show_shapes=True)

##save  agraph from tensorbaorda=
	


#model.load(MODEL_NAME)
#fig=plt.figure()

#
#print(model_1)

#y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
#y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
#confusion_matrix(y_actu, y_pred)



## initiliase a empty numoy array 
true_vals=[]
pred_vals=[]
prob_scores=[]

def predict_values(model=model_1):
    for num,data in enumerate(test_data[:]):
        # cat: [1,0]
        # dog: [0,1]dircd 

        
        #print(data)
        img_num = data[1]
        img_data = data[0]
        #print(img_num)
        #print(img_data)
        #y = fig.add_subplot(4,4,num+1)
        orig = img_data
        data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,1)# Rehsape the daya to be supllied correctly into tensorflow

        model_out = model.predict([data])[0]  ### produce a label of predicitions 
        



        if np.argmax(img_num) == 1: true_vals.append(0) 
        if np.argmax(img_num)==1: prob_scores.append(model_out[1])  ## return the index which is the maximum argument :
        if np.argmax(img_num) == 0: true_vals.append(1) 
        if np.argmax(img_num)==0: prob_scores.append(model_out[0])

        if np.argmax(model_out) == 1: pred_vals.append(0)  ## return the index which is the maximum argument :
        elif np.argmax(model_out) == 0: pred_vals.append(1)

    return (np.array(true_vals),np.array(pred_vals),np.array(prob_scores))        

#confusion_matrix



a=predict_values(model=model_1)

print("Accuracy ",accuracy_score(a[0], a[1]))
print("precision:",precision_score(a[0], a[1],average="weighted"))
print("roc_auc_score:",roc_auc_score(a[0], a[2]))
print("Recall:",recall_score(a[0], a[1], average='weighted'))
print(confusion_matrix(a[0], a[1]))

    



fpr, tpr, threshold = roc_curve(a[0], a[2])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
