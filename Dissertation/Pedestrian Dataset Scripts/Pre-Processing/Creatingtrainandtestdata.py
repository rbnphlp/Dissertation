import numpy as np
import os
import os.path
import shutil
import cv2
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

### Seperates the data into folders with background , and no of pedestrians
## read in a hthe labels of the pedestrian -datatset ##

#tf.reset_default_graph() #reset the graph after ecreating new model :

#backdoor_file= "C:/Users/creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian detection dataset/pedestrian detection dataset/backdoor/gt.txt"

#backdoor_folder_path="C:/Users/creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian detection dataset/pedestrian detection dataset/backdoor/in#put"

## seperate images for all the folders in the pedestrian datset:


## Replace with the folder name for the full pedestrian dataset :

big_folder="C:/Users/creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian detection dataset/pedestrian detection dataset"


IMG_SIZE=50
LR=1*10^(-3)


# Call thename for a model

## Name the model
MODEL_NAME ='ped-class-{}-{}.model'.format(LR,'Preprocessed- 2layer cifar structure -Dropout deactivated-ZCA whitening-norotation ')


def text_reader(file):
	#initialise a empty list
	lab=[]
	# read fille
	labels= open(file, "r")
	lines=labels.readlines()
## print each line
	for line  in lines:
		parts=line.split()
		#get the first column of the line

		if (len(parts)!=0):
			col1=parts[0]
			lab.append(int(col1))
		#print(col1)
		#lab1=lab.append(col1)


	return (lab)

#def load_images_numpy :


def  get_index_for_pedestrians(folder_path,gt_file):
	## Get the image label and split it with its index accoriding
	# to if the label comes


#folder_path = "test"
	images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
	gtlabels=text_reader(file=gt_file)
	#print(gtlabels)
	#print('images',images)

	## get the image index for each of the images
	for image in images:

		folder_name=image.split('n')[0]
		file_index=image.split('n')[1]
		file_index=file_index.split('.')[0]
		## remove the leading zeroes
		file_index=file_index.lstrip("0")
		image_index=int(file_index)

		if image_index in gtlabels:
			#rename the pedestrian pictures :
			print("renaming pedestrian Labels")
			os.rename(os.path.join(folder_path,image),os.path.join(folder_path,"ped"+str(image_index))+'.jpg')
			#ped.append(image_index)

		else :
		#print("Background images:",image_index)
			print("renaming background labels")
			os.rename(os.path.join(folder_path,image),os.path.join(folder_path,"no-ped"+str(image_index)+'.jpg'))
			#no_ped.append(image_index)


	#return ( {"ped":np.array(ped),"background":np.array(no_ped)})


def label_images_for_pedestrians(img):
	#function to label images for classification according to no background and pedestrians .

	 word_label=img.split('.')[-2]
	 word_label=word_label.split('/')[-1][:2]
	 #print(word_label)
	 if word_label=="no":return([0,1]) # no pedestrian
	 elif word_label=="pe":return([1,0]) #pedestrian


def create_train_data(TRAIN_DIR):
	training_data=[]
	for img in tqdm(os.listdir(TRAIN_DIR)):		# iterate through each of the images in the folder
		label=label_images_for_pedestrians(img)
		path=os.path.join(TRAIN_DIR,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img),np.array(label)])
		#print(training_data)
	shuffle(training_data)
	#np.save("train_data.npy",training_data) -- do not save the np array here ,save it after all the 6 iterations
	return training_data



def create_test_data(TEST_DIR):
	testing_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		label=label_images_for_pedestrians(img)
		img_num=img.split('.')[0] # get the id for each test for future refrence # need to compare with gt labels .
		path=os.path.join(TEST_DIR,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img),np.array(label)])
	#np.save("testdata.npy",testing_data) # no need to label test data
	shuffle(testing_data)
	return testing_data

def train_test_label(function="train"):
## Wrapper function to test /train and label the data
	if function=="train":
		## create an empty numpy array
		final_train_dataset=np.empty((0,2),int)# data for the full 6 folders
		for i in os.listdir(big_folder)[:9]:

				path=os.path.join(big_folder,i)
				file=os.path.join(path,'gt.txt')

				#print(file)
				image_path=os.path.join(path,"input")			#print(file)
				#print(image_path)
				train_data=create_train_data(image_path)

				final_train_dataset=np.append(final_train_dataset,train_data,axis=0)

		np.save("traindata.npy",final_train_dataset)
	elif function=="test":

	# create test data fro the remaining folders
		final_test_dataset=np.empty((0,2),int)
		for i in os.listdir(big_folder)[9:]:

				path=os.path.join(big_folder,i)
				file=os.path.join(path,'gt.txt')

				#print(file)
				image_path=os.path.join(path,"input")			#print(file)

				test_data=create_test_data(image_path)
				final_test_dataset=np.append(final_test_dataset,test_data,axis=0)

		np.save("testdata.npy",final_test_dataset)
	elif function=="rename":
		for i in os.listdir(big_folder):

				path=os.path.join(big_folder,i)
				file=os.path.join(path,'gt.txt')

				#print(file)
				image_path=os.path.join(path,"input")			#print(file)

				get_index_for_pedestrians(folder_path=image_path,gt_file=file)

	else :
		print("Type either test/train /rename")


## Folders in my current directory  after converted numpy and labelling

# create a property file and read me file

#train_data="C:/Users/creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/traindata.npy"
#test_data="C:/Users/creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/testdata.npy"
## Load them :

#train_data=np.load(train_data)
#test_data=np.load(test_data)
## function to  create folders according to n pedestrains:

#

if __name__=='__main__':



# Uncomment below to test or train or rename the data

	train_test_label("train")   # produce train data 
	train_test_label("test")     # produce test data

## Load them :
 
	train_data=np.load("traindata.npy")
	test_data=np.load(test_data)
	print(train_data.shape) #check if it worked ...

	