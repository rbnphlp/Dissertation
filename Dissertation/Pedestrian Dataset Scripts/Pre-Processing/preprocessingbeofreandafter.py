## Script to show a single picture beofre and after each pre-processing :

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
#from visualiselayers import * 

dir_to_save="C:/Users/creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/finalmodel/kerasmodels/preprocessedimage"
IMG_SIZE=50
train_data="C:/Users/Creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/traindata.npy"
test_data="C:/Users/Creem//Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/testdata.npy"

train_data=np.load(train_data)
test_data=np.load(test_data)    



def get_pedestrian(lab=0):
    ped_image_data=[]
        #function to get a pedestrian picture
    for  img,data in enumerate(train_data[:]):
          label=data[1]
          img_data=data[0]
          if np.argmax(label)==lab: 
            ped_image_data.append(img_data) 
    return(np.array(ped_image_data))


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=2.0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images



x=get_pedestrian()[1].reshape(-1,IMG_SIZE,IMG_SIZE,1)


# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:/Users/creem/Dropbox/Dissertation/Dissertation/FinalDataset-Change/pedestrian classification script/modelforconfusionmatrix/finalmodel/kerasmodels/preprocessedimage', save_prefix='pedestrian', save_format='jpeg'):
    i += 1
    if i > 5:
        break  # otherwise the generator would loop indefinitely