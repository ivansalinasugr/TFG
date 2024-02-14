import cv2 
import numpy as np
import os,sys,gc,argparse,random,time,uuid
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

 
from SCD_Model import SCDModel
from SCD_Augmentor import DataGenerator
from sklearn.model_selection import train_test_split

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver,TelegramObserver

from numpy.random import seed	
from tensorflow import set_random_seed
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
src_root="/mnt/homeGPU/isalinas/FSCDnet_Keras/"
exp_root="/mnt/homeGPU/isalinas/FSCDnet_Keras/"

ex= Experiment("Experimento_Keras")
ex.add_source_file(src_root+"SCDnet/SCD_Model.py")
ex.add_source_file(src_root+"SCDnet/SCD_Augmentor.py")
ex.add_source_file(src_root+"SCDnet/SCD_Main.py")
# ex.add_source_file(src_root+"SCDnet/SCD_Main_Tuner.py")
# ex.add_source_file(src_root+"SCDnet/SCD_Tuner.py")

ex.observers.append(FileStorageObserver.create(exp_root+'Experimento_Keras_runs'))
   
telegram_obs = TelegramObserver.from_config(src_root+'SCDnet/telegram.json')
ex.observers.append(telegram_obs)

def init_filestructure(hparams):
	if not os.path.exists(hparams['root_path']):
		os.mkdir(hparams['root_path']) 
	if not os.path.exists(hparams['root_path']+'/model_weights'):
		os.mkdir(hparams['root_path']+'/model_weights') 
	if not os.path.exists(hparams['root_path']+'/artifacts'):
		os.mkdir(hparams['root_path']+'/artifacts') 
	if not os.path.exists(hparams['root_path']+'/logs'):
		os.mkdir(hparams['root_path']+'/logs') 
		
def base_conf(): 
	hparams = { 
		'focal':27, #27.0 33.9 35.4 52.1 54.4 83.6
		#''' Train Parameters '''
		'GPUs': "0", # Tensorflow run configuration
		'n_epochs' : 10,  # cutoff the training after this many epochs
		'batch_size' :  32,	# size of the train batch 
		'val_split' : 0.2 , # portion of the data that will be used for training
		'steps_per_epoch': 224,
		'validation_steps':111,
		'validation_batch_size' : 32, # size of the validation batch		
		#''' Network Parameters '''
		'imgaug_seed': 1702, # Seed for image augmentation
		'padToFixedH':224, # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		'padToFixedW':224, # Size of subject images in the intermediate step of imgaug to fit background resolution (Width		
		#''' Network Parameters '''
		'patience' : 3,  # learning rate will be reduced after this many epochs if the validation loss is not improving
		'early_stop' : 6,  # training will be stopped after this many epochs without the validation loss improving
		'initial_learning_rate' : 0.0005,#f35 f85 # initial learning rate
		#'initial_learning_rate' :0.0005, #f27 f55
		'minimum_learning_rate' : 1e-12, # minimum value for the learning rate after reduction
		'learning_rate_drop' :  0.5,  # factor by which the learning rate will be reduced
		'with_dropout' : False, # Include a dropout layer if overfitting
		'dropout':0.5, # amount of dropout
		'with_batchnormalization' : False, # Include a BatchNormalization layer if overfitting
		'metrics': ['mean_absolute_error','mean_absolute_percentage_error'], #['mse','mae'], #Metrics
		'loss':'distortloss', # loss function
		'optimizer':'adam',
		'size_fc2':4096,
		#''' General Data Parameters '''
		'name': "NRIPS", # Name of the dataset
		'predictions_file': os.path.abspath(exp_root+"submission.csv"),
		'overwrite' : True,  # If True, will previous files. If False, will use previously written files.
		#'data_file' : os.path.abspath("Analand_data.h5"),
		#'root_path' :os.path.abspath("./results"), #Root folder to store, load and perform operations
		#'model_file' : os.path.abspath("./model_weights.hdf5"),
		#'dir' : "./_data_/", #Root folder whre h5 and images are stored
		#'file': "./_data_/CMDP/CMDP.h5", # route to the HDF5 file with the structure to be used
		'backbone': 'vgg',#'vgg',# vgg, resnet, inception
		'lastConv':True,
		'image_shape' : (224, 224, 3),  # This determines what shape the images will be cropped/resampled to.  
		#'image_shape' : (224, 224, 3),  # This determines what shape the images will be cropped/resampled to.  
		'load_model' : False,
		'load_weight_model':False,
		'debug':True,
		'expand_model':True
	}
	hparams['uuid']=str(uuid.uuid4())[:5]
	hparams['root_path']=os.path.abspath(exp_root+ "F" +str(hparams['focal']))
	hparams['model_file']=os.path.abspath(hparams['root_path']+"/model_weights/model-weights_f"+str(hparams['focal'])+"_"+hparams['uuid']+".hdf5")
	
	#hparams['input_model_file']=os.path.abspath(exp_root+"/deploy_models/BEST_f27_37161.hdf5")
	#hparams['input_model_file']=os.path.abspath(exp_root+"/deploy_models/BEST_f35_390f5.hdf5")
	#hparams['input_model_file']=os.path.abspath(exp_root+"/deploy_models/BEST_f55_b910c.hdf5")
	#hparams['input_model_file']=os.path.abspath(exp_root+"/deploy_models/BEST_f85_24bb7.hdf5")


	hparams['training_file'] = os.path.abspath(src_root+"input/train_labels.csv")
	hparams['testing_file'] = os.path.abspath(src_root+"input/test_labels.csv")

	return hparams

@ex.config
def my_hparams():
	hparams=base_conf()
    
# @ex.named_config
# def variant1():	
#     hparams=base_conf()
#     hparams["n_epochs"]=100
    
def purgeData(data,focal,fraction=1):
	if(focal==35):
		purged_data=data[data['Focal'].isin([32.1,33.9,33,35,35.4])]
	elif(focal==55):
		purged_data=data[data['Focal'].isin([52.1,53,53.5,54.4,55])]
	elif(focal==85):
		purged_data=data[data.Focal.isin([83.6,84.1])]
	else:
		purged_data=data[data['Focal']==focal] 
	#purged_data=purged_data[purged_data.Distance!=100]
	#Now we sample a % of the data for validation purposes
	print("origin",len(purged_data))
	purged_data=purged_data.sample(frac=fraction)
	print("sampled",len(purged_data))

	return purged_data

def loadData(db,hparams,  val_split=0.2, sub_sample_size=-1):
	#backgrounds = glob(os.path.join(hparams['backgrounds'], '*.jpg'))
	#'Loads data into generator object'
	if db == "train":
		data = pd.read_csv(hparams['training_file'],sep=',')
		data=purgeData(data,hparams['focal'])
		image_data=data.loc[:, data.columns != 'Distance'][:sub_sample_size]
		labels = data['Distance'].values[:sub_sample_size]
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
		data=purgeData(data,hparams['focal'])
		image_data =data.loc[:, data.columns != 'Distance'][:sub_sample_size]
		print("test images",len(image_data.index))
		labels = data['Distance'].values[:sub_sample_size]
		if sub_sample_size== -1 :
			return DataGenerator(image_data, labels, hparams, batch_size=1,augment=False ), None
		else:
			return DataGenerator(image_data[:sub_sample_size], labels[:sub_sample_size], hparams, batch_size=1,augment=False), None

@ex.automain
def main(_run,hparams):
	seed(1)
	set_random_seed(2)
	tf.compat.v1.set_random_seed(2)
	random.seed(3)
	#os.environ["CUDA_VISIBLE_DEVICES"] = hparams['GPUs']
	
	init_filestructure(hparams)
	# create model 

	model = SCDModel(hparams)
	if(hparams['debug']):
		model.summary()
		
	# train model
	train_data, val_data = loadData("train", hparams, val_split=hparams['val_split']  )
	#train_data, val_data = loadSavedData("train", hparams )
	#counter=0
	#for batch in train_data:
	#	counter+=1
	#	print(counter,"-batch, size:",len(batch),len(batch[0]),len(batch[0][0]),len(batch[0][0][0]),len(batch[0][0][0][0]) )
	#	for i in range (0,1):
	#		print(counter,"-batch, size:",len(batch),len(batch[0]),len(batch[0][0]),len(batch[0][0][0]),len(batch[0][0][0][0]) )
	#		image = batch[0][i]
	#		plt.imshow(image.astype('uint8'))
	#		plt.show()
  
  
#	train_data, val_data = loadData("train", hparams, val_split=hparams['val_split']  )
	history=model.train(train_data, val_data, plot_results=False)
	val_metric=np.min(history['val_mean_absolute_error'])

	retloss=0
	for k in history.keys(): 
		for i in range(len(history[k])): 
			_run.log_scalar(k,history[k][i],i) 
	predictions=model.predict(val_data)
	predictions = [y for x in predictions for y in x]
	
	val_y=([])
	for k in range(0,len(val_data)):
		batch = val_data.__getitem__(k)
		val_y=np.append(val_y,batch[1])
	
	predictions_df = pd.DataFrame({"predictions": predictions, "targets": val_y})
	filename =hparams['root_path']+"/artifacts/predictions_df_"+hparams['uuid']+".pickle"
	predictions_df.to_pickle(filename)
	_run.add_artifact(filename, name="predictions_df", content_type="application/octet-stream")
		
	std=repr(np.std(val_y-predictions, axis= 0))
	p99=repr(np.percentile(val_y-predictions, 99, axis=0))
	dist_y=np.array([(1/(1 + x/0.1265)) for x in val_y])
	dist_pred=np.array([(1/(1 + x/0.1265)) for x in predictions])
	error=np.absolute(dist_y-dist_pred)
	distortion=repr(np.sum(error))
	_run.log_scalar('std',std)
	_run.log_scalar('p99',p99)
	_run.log_scalar('distortion',distortion)
 
	if(val_metric<7):
		print("Good Validation Result, we will test the model")
		test_data,none = loadData("test", hparams,sub_sample_size=-1   )
		test_predictions=model.predict(test_data)
		test_predictions = [y for x in test_predictions for y in x]
	
		test_y=([])
		for k in range(0,len(test_data)):
			batch = test_data.__getitem__(k)
			test_y=np.append(test_y,batch[1])
	
		test_predictions_df = pd.DataFrame({"predictions": test_predictions, "targets": test_y})
		test_filename =hparams['root_path']+"/artifacts/test_predictions_df_"+hparams['uuid']+".pickle"
		test_predictions_df.to_pickle(test_filename)
		_run.add_artifact(test_filename, name="test_predictions_df_"+hparams['uuid'], content_type="application/octet-stream")
		
		test_std=repr(np.std(test_y-test_predictions, axis= 0))
		test_p99=repr(np.percentile(test_y-test_predictions, 99, axis=0))
		test_dist_y=np.array([(1/(1 + x/12.6572)) for x in test_y])
		test_dist_pred=np.array([(1/(1 + x/12.6572)) for x in test_predictions])
		test_error=np.absolute(test_dist_y-test_dist_pred)
		test_distortion=repr(np.sum(test_error))
		_run.log_scalar('test_std',test_std)
		_run.log_scalar('test_p99',test_p99)
		_run.log_scalar('test_distortion',test_distortion)


	model.release()
	tf.compat.v1.reset_default_graph()
	tf.keras.backend.clear_session()

	del model,train_data,val_data
	gc.collect() 
	return distortion
	#Visualize bacth input
	
	# counter=0
	# for k in range(0,10):
	# 	for j in range(0,10):
	# 		batch = train_data.__getitem__(j)
			
	# 		counter+=1
	# 		print(counter,"-batch, size:",len(batch),len(batch[0]),len(batch[0][0]),len(batch[0][0][0]),len(batch[0][0][0][0]) )
			
	# 		###Batch = [2/3,batch_size,244,244,3] 
	# 		###Batch[0] = Image
	# 		# If we merge focals into input...
	# 		###Batch[1] = Focal 
	# 		###Batch[2] = Distance
	# 		# Else
	# 		###Batch[1] = Distance
	# 		print(batch[1])
	# 		for i in range (0,len(batch[0])):  
	# 			image = batch[0][i]
	# 		# 	print(batch[1][i],batch[2][i])
	# 			plt.imshow(image)
	# 			plt.show()
    # #model.train(train_data, val_data, plot_results=True)
	# 	train_data.on_epoch_end()	
    # submit model
    #test_data, _ = loadData("test", hparams,args.backgrounds)
    #model.create_submit(test_data)
