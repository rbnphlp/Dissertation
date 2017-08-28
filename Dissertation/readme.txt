The scripts  are arranged logically in relation to the report.


######################### Tensforflow using MNIST #####################################################################
Contains 4 files which runs the 4 models as specified in the report 

Model1
Model2
Model3
Model4





######################## Pedestrian Classficiation scripts #########################################################

* Pre-Processing Folder: 
		* Creatingtrainandtest : Stand alone file which produces the train and test numpy arrays.Need to download 
 		http://www.changedetection.net/ >> datasets >> Pedestrian detection dataset >>  predestrian detection dataset.zip and 
		change big_folder to the corresponding location .
		
		* preprocessingbeforeandafter : Takes a pedestrianimage from training set and applies image augmentation and pre-processing steps as 
		required and saves it to a directory .
					
		
* Data folder : Contains two files train & test. npy  images converted into arrays (results of the creatingtrainandtest Processing Folder)

* Model Evaluation folder :
		 
		
		* simualtion batchsizes : runs kerasmodel.py 100 times , saves the results to a txt file and then prints the results to a screen 
		* visualiselayers : visualises the layers for the model , either "bestmodel" & "badmodel" is default.
		
		* plotsforbatchsizes.r : an r files to use the results from simulation batchsizes to plot simulation curves 
		

* Testing Folder :
		* confusionmatrixandrocccurves : produces confusionmatrix and roc curves for the specified model
		
* cif10 -2layer model -Tflearn folder  :
		* cif102layer : Architecutre for the 2layer model 
		* traincif102layer : runs the architecutre of the 2layer
* cifar10finalmodel-training :
		* kerasmodel : runs the model with cifar10structure specified , tests it against test set and returns classification metrics
		* bestmodel.h5 & badmodel.h5 : saved models from keras can be used to predict on test data

