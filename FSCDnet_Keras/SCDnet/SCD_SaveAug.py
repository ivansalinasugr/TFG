import cv2 
import numpy as np
import os,sys,gc,argparse,random,time,uuid
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


from SCD_Augmentor import DataGenerator
from sklearn.model_selection import train_test_split
 
from numpy.random import seed	
from tensorflow import set_random_seed
from sacred import Experiment

#TODO: Test then adapt
#src_root="/home/HDD/SCD_Forensics/SCD_Database/" 
src_root="/mnt/homeGPU/ebermejo/SCDForensics/" 

ex= Experiment("IMGAug_SCD_Forensics")


def init_filestructure(hparams):
	if not os.path.exists(hparams['root_path']):
		os.mkdir(hparams['root_path'])   
	if not os.path.exists(hparams['root_path']+'/train'):
		os.mkdir(hparams['root_path']+'/train') 
	if not os.path.exists(hparams['root_path']+'/val'):
		os.mkdir(hparams['root_path']+'/val') 
  

@ex.config
def base_conf():
	hparams = { 
		'focal':35, #27.0 33.9 35.4 52.1 54.4 83.6
		'merge_focals':False,
		#''' Train Parameters '''
		'GPUs': "0", # Tensorflow run configuration
		'batch_size' :  64,	# size of the train batch 
		'epochs' : 150,
		'val_split' : 0.2 , # portion of the data that will be used for training
		'validation_batch_size' : 32, # size of the validation batch		
		#''' Network Parameters '''
		'imgaug_seed': 1702, # Seed for image augmentation
		'padToFixedH':768, # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		'padToFixedW':768, # Size of subject images in the intermediate step of imgaug to fit background resolution (Width		
		'regression' :  True,  # Switch to configure the network as regression / classification. This will require a change in the loss function and label structure (SCD_Augmentor)
        'normalize' : False,  # Switch to activate label/focal data normalization or not
		#''' General Data Parameters '''
		#'name': "NRIPS", # Name of the dataset
		'backgrounds':os.path.abspath(src_root+"rbgs/"),
		#'training_file' : os.path.abspath("./input/synth_train_labels.h5"),
		#'testing_file' : os.path.abspath("./input/test_labels.h5"),
		 
		'overwrite' : True,  # If True, will previous files. If False, will use previously written files.
		#'data_file' : os.path.abspath("Analand_data.h5"),
		#'root_path' :os.path.abspath("./results"), #Root folder to store, load and perform operations
		#'model_file' : os.path.abspath("./model_weights.hdf5"),
		#'dir' : "./_data_/", #Root folder whre h5 and images are stored
		#'file': "./_data_/CMDP/CMDP.h5", # route to the HDF5 file with the structure to be used
		'backbone': 'vgg',# vgg, resnet, inception
		'image_shape' : (244, 244, 3),  # This determines what shape the images will be cropped/resampled to.  
		#'image_shape' : (224, 224, 3),  # This determines what shape the images will be cropped/resampled to.  
		 
		'debug':False, 
	}
	hparams['uuid']=str(uuid.uuid4())[:5]
	hparams['root_path']=os.path.abspath(src_root+'/afullsynth_aug')
	 
	img_set='fullsynth'#'real'
	if(img_set=='zsynth'):
		hparams['padToFixedH']=384 # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		hparams['padToFixedW']=384 # Size of subject images in the intermediate step of imgaug to fit background resolution (Width)
		hparams['training_file'] = os.path.abspath(src_root+"input/zsynth_train_labels.csv")
		hparams['testing_file'] = os.path.abspath(src_root+"input/zsynth_test_labels.csv")
		hparams['name']= "Stirling"
	elif(img_set=='fullsynth'):
		hparams['padToFixedH']=384 # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		hparams['padToFixedW']=384 # Size of subject images in the intermediate step of imgaug to fit background resolution (Width)
		hparams['training_file'] = os.path.abspath(src_root+"input/Comsynth_train_labels.csv")
		hparams['testing_file'] = os.path.abspath(src_root+"input/Comsynth_test_labels.csv")
		hparams['name']= "StirlingFull"
	elif(img_set=='real'):
		hparams['padToFixedH']=768 # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		hparams['padToFixedW']=768 # Size of subject images in the intermediate step of imgaug to fit background resolution (Width)
		hparams['training_file'] = os.path.abspath(src_root+"input/Last_train_labels.csv")
		hparams['testing_file'] = os.path.abspath(src_root+"input/Last_test_labels.csv")
		hparams['name']= "Real"
	else:
		hparams['padToFixedH']=768 # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		hparams['padToFixedW']=768 # Size of subject images in the intermediate step of imgaug to fit background resolution (Width)
		hparams['training_file'] = os.path.abspath(src_root+"input/train_labels.h5")
		hparams['testing_file'] = os.path.abspath(src_root+"input/test_labels.h5")
		hparams['name']= "NRIPS" 

   
#Method for selecting only the data associated for a determined focal
def purgeData(data,focal):
	if(focal==35):
		purged_data=data[data['Focal'].isin([32.1,33.9,33,35,35.4])]
	elif(focal==55):
		purged_data=data[data['Focal'].isin([52.1,53,53.5,54.4,55])]
	elif(focal==85):
		purged_data=data[data.Focal.isin([83.6,84.1])]
	else:
		purged_data=data[data['Focal']==focal] 
	#purged_data=purged_data[purged_data.Distance!=100]
	return purged_data

def loadData(db,hparams,  val_split=0.2, sub_sample_size=-1):
	#backgrounds = glob(os.path.join(hparams['backgrounds'], '*.jpg'))
	#'Loads data into generator object'
	if db == "train":
		data = pd.read_csv(hparams['training_file'],sep=',')
		if(not hparams['merge_focals']):
			data=purgeData(data,hparams['focal'])
		image_data=data.loc[:, data.columns != 'Distance'][:sub_sample_size]
		labels = data['Distance'].values[:sub_sample_size]
		print("train images: "+str(len(image_data)))
		if val_split > 0:
			X_train, X_test, Y_train, Y_test = train_test_split(image_data, labels, test_size=val_split) 
			print("train images: "+str(len(X_train))+" val images: "+str(len(X_test)))
			train_data = DataGenerator(X_train, Y_train,  hparams,  batch_size=hparams['batch_size'], augment=True, shuffle=True)
			val_data = DataGenerator(X_test, Y_test, hparams,  batch_size=hparams['validation_batch_size'], augment=False, shuffle=False) 
			return train_data, val_data
		else:
			return DataGenerator(image_data, labels, hparams,  batch_size=hparams['batch_size'], augment=True, shuffle=True), None
	else:
		data = pd.read_csv(hparams['testing_file'],sep=',')
		if(not hparams['merge_focals']):
			data=purgeData(data,hparams['focal'])
		image_data =data.loc[:, data.columns != 'Distance'][:sub_sample_size]
		print("test images",len(image_data.index))
		labels = data['Distance'].values[:sub_sample_size]
		if sub_sample_size== -1 :
			return DataGenerator(image_data, labels, hparams, batch_size=2,augment=False ), None
		else:
			return DataGenerator(image_data[:sub_sample_size], labels[:sub_sample_size], hparams, batch_size=1,augment=False), None

def new_config(cdict,focal):
	config=cdict['hparams'] 
	for (key, value) in cdict.items(): 
		if(type(value)!=dict):
			config[key]=cdict[key]
		if(key=='dp'):
			for (key2, value2) in value.items(): 
				config[key2]=value2
	config['focal']=focal
	return config

@ex.automain
def main(_run,hparams):
	seed(1)
	set_random_seed(2)
	tf.compat.v1.set_random_seed(2)
	random.seed(3) 
	init_filestructure(hparams)
	cset="train"
	for focal in [35]:
		txt_id= os.path.abspath(hparams['root_path']+'/'+cset+'_datalist_'+str(focal)+'.csv')
		txt_val_id= os.path.abspath(hparams['root_path']+'/validation_datalist_'+str(focal)+'.csv')
		ftxt=open(txt_id,'w')
		ftxt.write("Path,focal,epoch,counter,batch,distance\n")
		ftxtval=open(txt_val_id,'w')
		ftxtval.write("Path,focal,epoch,counter,batch,distance\n")
		config=new_config(_run.config,focal)
		train_data, val_data = loadData(cset, config, val_split=hparams['val_split']  )
		for e in range(0,hparams['epochs']): #CHANGE TO ESTIMATED EPOCHS
			counter=0
			for batch in train_data:
				counter+=1
				print(e," Batch-",counter," size:",len(batch),len(batch[0]),len(batch[0][0]),len(batch[0][0][0]),len(batch[0][0][0][0]) )

				for i in range (0,len(batch[1])):  
					image = batch[0][i]  
					label = batch[1][i]
					img_id= os.path.abspath(hparams['root_path']+'/'+cset+'/ia'+cset+'_'+str(focal)+'_'+str(e)+'_'+str(counter)+'_'+str(i)+'_'+str(label)+'.jpg')
					cv2.imwrite(img_id, image)
					ftxt.write(img_id+','+str(focal)+','+str(e)+','+str(counter)+','+str(i)+','+str(label)+'\n')
					#plt.imshow(image)
					#plt.show() 
			if(cset=="train"):
				if(val_data):
					for batch in val_data:
							csetv="val"
							counter+=1
							print(e," Batch-",counter," size:",len(batch),len(batch[0]),len(batch[0][0]),len(batch[0][0][0]),len(batch[0][0][0][0]) )

							for i in range (0,len(batch[1])):  
								image = batch[0][i]  
								label = batch[1][i]
								img_id= os.path.abspath(hparams['root_path']+'/'+csetv+'/ia'+csetv+'_'+str(focal)+'_'+str(e)+'_'+str(counter)+'_'+str(i)+'_'+str(label)+'.jpg')
								cv2.imwrite(img_id, image)
								ftxtval.write(img_id+','+str(focal)+','+str(e)+','+str(counter)+','+str(i)+','+str(label)+'\n')
								#plt.imshow(image)
								#plt.show()
     
     
     
		ftxt.close()
		ftxtval.close()
			#	break;
					#TODO:
				###   Change input path from external hard drive
				#   Change load_data for a load_saved_data where we can get both train and val sets of the SCD_tuning
				#   Test new VGG implementation, then Resnet, then Inception.
				#   Run new experiments.
