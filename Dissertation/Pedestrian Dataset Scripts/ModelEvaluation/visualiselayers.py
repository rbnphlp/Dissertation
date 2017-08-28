''' Script to look at how the model differentiates with a pedestrian between a good model and worse performing model'''



'''
Code borrowed from https://gist.github.com/hadim/9fedb72b54eb3bc453362274cd347a6a
'''
## get a pedestrian image and pass it as a image to both the models , see 

#from labelimagesconfustionmatrix import * 
#from confusionmatrixkerasmodel import * 
from keras.models import Model

import numpy as np
from keras.models import Model
import keras.backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import load_model

IMG_SIZE =50 

## change for data structure :


train_data="C:/Users/phili/Desktop/zca/traindata.npy"
test_data="C:/Users/phili/Desktop/zca/testdata.npy"
train_data=np.load(train_data)
test_data=np.load(test_data)    

model_1="bestmodel.h5"
model_1 = load_model(model_1)
model_2="badmodel.h5"
model_2 = load_model(model_2)

def get_pedestrian(lab=0):
    ped_image_data=[]
        #function to get a pedestrian picture
    for  img,data in enumerate(train_data[:]):
          label=data[1]
          img_data=data[0]
          if np.argmax(label)==lab: 
            ped_image_data.append(img_data) 
    return(np.array(ped_image_data))


def visualize_pedestrian(model, pedestrian,layer):
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    pedestrian_batch = pedestrian.reshape(-1,IMG_SIZE,IMG_SIZE,1)
    ##Choose the right no of layers
    #layer_name = 'my_layer'
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[layer].output)
    conv_pedestrian = intermediate_layer_model.predict(pedestrian_batch)
    print(conv_pedestrian)
    conv_pedestrian = np.squeeze(conv_pedestrian, axis=0)
    print(conv_pedestrian.shape)
    plt.imshow(conv_pedestrian)
    plt.show()




def make_mosaic(im, nrows, ncols, border=1):

    ## helper function for the below visualisations###
    """From http://nbviewer.jupyter.org/github/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    """
    import numpy.ma as ma

    nimgs = len(im)
    imshape = im[0].shape
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    im
    for i in range(nimgs):
        
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = im[i]
        
    return mosaic
## Pass a single ped image and see how the final layer looks like for the working model  :
#visualize_pedestrian(model=model_1,pedestrian=get_pedestrian()[1],layer=1)
def plot_feature_map(model, layer_id, X, n=256, ax=None, **kwargs):
    """

    need to author code
    """
    import keras.backend as K
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    layer = model.layers[layer_id]
    
    try:
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
        activations = get_activations([X, 0])[0]
    except:
        # Ugly catch, a cleaner logic is welcome here.
        raise Exception("This layer cannot be plotted.")
        
    # For now we only handle feature map with 4 dimensions
    if activations.ndim != 4:
        raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name,
                                                                                             activations.ndim))
        
    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "jet"
        
    fig = plt.figure(figsize=(15, 15))
    
    # Compute nrows and ncols for images
    n_mosaic = len(activations)
    nrows = int(np.round(np.sqrt(n_mosaic)))
    ncols = int(nrows)
    if (nrows ** 2) < n_mosaic:
        ncols +=1
        
    # Compute nrows and ncols for mosaics
    if activations[0].shape[0] < n:
        n = activations[0].shape[0]
        
    nrows_inside_mosaic = int(np.round(np.sqrt(n)))
    ncols_inside_mosaic = int(nrows_inside_mosaic)

    if nrows_inside_mosaic ** 2 < n:
        ncols_inside_mosaic += 1

    for i, feature_map in enumerate(activations):

        mosaic = make_mosaic(feature_map[:n], nrows_inside_mosaic, ncols_inside_mosaic, border=1)

        ax = fig.add_subplot(nrows, ncols, i+1)
        
        im = ax.imshow(mosaic, **kwargs)
        ax.set_title("Feature map #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,
                                                                                  layer.name,
                                                                                  layer.__class__.__name__))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
            
    fig.tight_layout()
    plt.show()
    return fig
    #
    #

def plot_all_feature_maps(model, X, n=256, ax=None, **kwargs):
    """
    """
    
    figs = []
    
    for i, layer in enumerate(model.layers):
        
        try:
            fig = plot_feature_map(model, i, X, n=n, ax=ax, **kwargs)
        except:
            pass
        else:
            figs.append(fig)
            
    return figs

if __name__=="__main__":

    #img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
  
    reshaped_image=get_pedestrian(lab=0)[1].reshape(-1,IMG_SIZE,IMG_SIZE,1)
    plt.imshow(get_pedestrian(lab=0)[1])
    plot_all_feature_maps(model_1,X=reshaped_image ,n=256)
    plot_all_feature_maps(model_2,X=reshaped_image,n=256)


