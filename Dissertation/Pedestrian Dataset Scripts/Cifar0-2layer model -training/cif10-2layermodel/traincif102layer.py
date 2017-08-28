## Script to run to train the model after pre-processed data 

#from  Creatingtrainandtestdata  import *
#from pedcovnet6conf import * 
from  cif102layer import * 
import matplotlib.pyplot as plt
#raph =tf.Graph()
import sklearn.model_selection as sk

batch_size=128

## change the path accordingly..

train_data="C:/Users/phili/Desktop/zca/traindata.npy"
test_data="C:/Users/phili/Desktop/zca/testdata.npy"

train_data=np.load(train_data)
test_data=np.load(test_data)



## Run the script again to keep adding more epochs 
# test=train_data[17000:]
# train=train_data[:17000]
# X= np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)## get the 1st element which is the image and the 2nd element to ge thte labels
# Y=[i[1]for i in train ] # fit the labels for the Y daata


X= np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)## get the 1st element which is the image and the 2nd element to ge thte labels
Y=[i[1]for i in train_data ] # fit the labels for the Y daata

X_train, test_X, y_train, test_Y = sk.train_test_split(X, Y, test_size=0.2)

X_train = X
y_train = Y

#
# test_X = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
# test_Y=[i[1]for i in test]

if __name__=="__main__":

	#if os.path.exists('{}.meta'.format(MODEL_NAME)):
	#	model.load(MODEL_NAME)
	#	print("model loaded !")

	

		

	#model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': test_X}, {'targets': test_Y}), 
    #snapshot_step=500, show_metric=True)

	fig=plt.figure()
	
	for num,data in enumerate(test_data[:16]):
	    # cat: [1,0]
	    # dog: [0,1]dircd 
	 
	     
	    #print(data)
	    img_num = data[1]
	    img_data = data[0]
	    #print(img_num)
	    #print(img_data)
	    y = fig.add_subplot(4,4,num+1)
	    orig = img_data
	    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1) # Rehsape the daya to be supllied correctly into tensorflow
	 
	    #model_out = model.predict([data])[0]
	    #print(model.predict([data]))
	    model_out = model.predict([data])[0]
	    #print(model_out)
	     
	    if np.argmax(model_out) == 1: str_label='No Pedestrian'  ## return the index which is the maximum argument :
	    else: str_label='Pedestrian'
	         
	    y.imshow(orig,cmap='gray')
	    plt.title(str_label)
	    y.axes.get_xaxis().set_visible(False)
	    y.axes.get_yaxis().set_visible(False)
	plt.show()




		#modelout.append(model_out)

#print(modelout)


	
