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
from hyperopt import  STATUS_OK
from numpy.random import seed	 
from tensorflow import set_random_seed 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

src_root="/mnt/homeGPU/ebermejo/SCDForensics/"
exp_root="/mnt/homeGPU/ebermejo/SCDForensics/"


ex= Experiment("Tune_35ASCD_RN_Detection")

ex.add_source_file(src_root+"SCDnet/SCD_Model.py")
ex.add_source_file(src_root+"SCDnet/SCD_Augmentor.py")
ex.add_source_file(src_root+"SCDnet/SCD_Main.py")
ex.add_source_file(src_root+"SCDnet/SCD_Main_Tuner.py")
ex.add_source_file(src_root+"SCDnet/SCD_Tuner.py")

#ex.observers.append(MongoObserver.create(
#    url='mongodb+srv://sacred:j4sffvxhDd2Kl1O9@sacredugr-8ggnv.mongodb.net/test?retryWrites=true&w=majority',
#    db_name='Tune35_metric_SCD_ForensicsDB'))
ex.observers.append(FileStorageObserver.create(exp_root+'Tune35_ASCD_RN_runs'))
   
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

@ex.config
def base_conf(): 
	#Specific parameters for activating and tuning dropout
	dp={'with_dropout' : False, # Include a dropout layer if overfitting
		'dropout':0.5, # amount of dropout
		'with_batchnormalization':False
		}
	#General parameters
	hparams = { 
		'debug':False,
		'focal':35, #27.0 33.9 35.4 52.1 54.4 83.6
		'merge_focals':False,
		#''' Train Parameters '''
		'GPUs': "0", # Tensorflow run configuration
        'n_epochs' : 150,  # cutoff the training after this many epochs
        'batch_size' :  64,	# size of the train batch 
        'val_split' : 0.2 , # portion of the data that will be used for training
        'validation_batch_size' : 32, # size of the validation batch

		#''' Network Parameters '''
		'imgaug_seed': 1702, # Seed for image augmentation


		#''' Network Parameters '''
		'image_shape' : (244, 244, 3),  # This determines what shape the images will be cropped/resampled to.  
        'patience' : 2,  # learning rate will be reduced after this many epochs if the validation loss is not improving
        'early_stop' : 6,  # training will be stopped after this many epochs without the validation loss improving
        'initial_learning_rate' :10e-6, # initial learning rate
        'minimum_learning_rate' : 10e-10, # minimum value for the learning rate after reduction
        'learning_rate_drop' :  0.5,  # factor by which the learning rate will be reduced
		'with_dropout' : False, # Include a dropout layer if overfitting
        'dropout':0.5, # amount of dropout
		'with_batchnormalization' : False, # Include a BatchNormalization layer if overfitting
        'regression' :  True,  # Switch to configure the network as regression / classification. This will require a change in the loss function and label structure (SCD_Augmentor)
        'normalize' : True,  # Switch to activate label/focal data normalization or not
		'metrics': ['mean_absolute_error'], #['mse','mae'], #Metrics
		'loss':'mse', # loss function
		'optimizer':'adam',
		#''' General Data Parameters '''
		'name': "NRIPS", # Name of the dataset
		'backgrounds':os.path.abspath(exp_root+"rbgs/"),
        'training_file' : os.path.abspath(src_root+"input/train_labels.h5"),
        'testing_file' : os.path.abspath(src_root+"input/test_labels.h5"),
        'training_aug_file' : os.path.abspath(src_root+"input/train_datalist_labels.h5"),
        'validation_aug_file' : os.path.abspath(src_root+"input/test_datalist_labels.h5"),
		'load_model': False,
		'load_weight_model':False,
        'overwrite' : True,  # If True, will previous files. If False, will use previously written files.
		'expand_model':True,
		'input_model_file':os.path.abspath(exp_root+"/deploy_models/N500_besti2_f27_84e09.hdf5"),



        #'data_file' : os.path.abspath("Analand_data.h5"),
        #'root_path' :os.path.abspath("./results"), #Root folder to store, load and perform operations
        #'model_file' : os.path.abspath("./model_weights.hdf5"),
        #'dir' : "./_data_/", #Root folder whre h5 and images are stored
        #'file': "./_data_/CMDP/CMDP.h5", # route to the HDF5 file with the structure to be used
		'backbone': 'vgg',# vgg, resnet, inception
		'lastConv':False

    }

	img_set='fullsynth'#'real'
	if(img_set=='zsynth'):
		hparams['padToFixedH']=384 # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		hparams['padToFixedW']=384 # Size of subject images in the intermediate step of imgaug to fit background resolution (Width)
		hparams['training_file'] = os.path.abspath(src_root+"input/zsynth_train_labels.h5")
		hparams['testing_file'] = os.path.abspath(src_root+"input/zsynth_test_labels.h5")
		hparams['training_aug_file'] = os.path.abspath(src_root+"asynth_aug/train_datalist_"+str(hparams['focal'])+".csv")
		hparams['validation_aug_file'] = os.path.abspath(src_root+"asynth_aug/validation_datalist_"+str(hparams['focal'])+".csv")
		hparams['name']: "Stirling"
	elif(img_set=='fullsynth'):
		hparams['padToFixedH']=384 # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		hparams['padToFixedW']=384 # Size of subject images in the intermediate step of imgaug to fit background resolution (Width)
		hparams['training_file'] = os.path.abspath(src_root+"input/Comsynth_train_labels.csv")
		hparams['testing_file'] = os.path.abspath(src_root+"input/Comsynth_test_labels.csv")
		hparams['training_aug_file'] = os.path.abspath(src_root+"afullsynth_aug/train_datalist_"+str(hparams['focal'])+".csv")
		hparams['validation_aug_file'] = os.path.abspath(src_root+"afullsynth_aug/validation_datalist_"+str(hparams['focal'])+".csv")
		hparams['name']= "StirlingFull"
	else:
		hparams['padToFixedH']=768 # Size of subject images in the intermediate step of imgaug to fit background resolution (Height)
		hparams['padToFixedW']=768 # Size of subject images in the intermediate step of imgaug to fit background resolution (Width)
		hparams['training_file'] = os.path.abspath(src_root+"input/Last_train_labels.h5")
		hparams['testing_file'] = os.path.abspath(src_root+"input/Last_test_labels.h5")
		hparams['training_aug_file'] = os.path.abspath(src_root+"areal_aug/train_datalist_"+str(hparams['focal'])+".csv")
		hparams['validation_aug_file'] = os.path.abspath(src_root+"areal_aug/validation_datalist_"+str(hparams['focal'])+".csv")
		hparams['name']: "Real" 
	#Tunable hyperparameters, They will overwrite those inside hparams. Needed here for hyperopt to overwrite their values inside sacred
	batch_size=64# size of the train batch  
	focal=35
	n_epochs=150
	initial_learning_rate=10e-6
	minimum_learning_rate=10e-10
	patience=2
	early_stop=6
	optimizer='adam'
	loss='mse'
	normalize=False
	size_fc2=4096  	
	uuid='a1234'
	backbone='vgg'
	lastConv=False
	

#Method for selecting only the data associated for a determined focal
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
			return DataGenerator(image_data, labels, hparams, batch_size=1,augment=False ), None
		else:
			return DataGenerator(image_data[:sub_sample_size], labels[:sub_sample_size], hparams, batch_size=1,augment=False), None


def loadSavedData(db,hparams):
    #TODO: Test if this works
	if db == "train":
		train_data = pd.read_csv(hparams['training_aug_file'])
		val_data = pd.read_csv(hparams['validation_aug_file'])
		tdatagen=tf.keras.preprocessing.image.ImageDataGenerator()
		vdatagen=tf.keras.preprocessing.image.ImageDataGenerator()
		train_generator=tdatagen.flow_from_dataframe(
			dataframe=train_data,
			#directory=os.path.abspath(src_root+"synth_aug/train/"),
			x_col="Path",
			y_col="distance",
			#subset="training",
			batch_size=hparams['batch_size'],
			seed=2021,
			shuffle=True,
			class_mode="raw",
			target_size=(244,244))
		val_generator=vdatagen.flow_from_dataframe(
			dataframe=val_data,
			#directory=os.path.abspath(src_root+"synth_aug/val/"),
			x_col="Path",
			y_col="distance",
			#subset="validation",
			batch_size=hparams['validation_batch_size'],
			seed=2021,
			shuffle=True,
			class_mode="raw",
			target_size=(244,244))
		return train_generator,val_generator
  

#Objective function for hyperopt tuning
def hyper_objective(params): 
	config = {} 
	if type(params) == dict:
		params = params.items()

	for (key, value) in params: 
		config[key] = value
	start_time = time.time()
	run = ex.run(config_updates=config)
	err = run.result
	return {'loss': err, 'hyperparams':config, 'eval_time': time.time() - start_time, 'uuid': run.config['uuid'],'status': STATUS_OK}

#Function to overwrite all the tuning parameters into a single dictionary
def tune_config(cdict):
	config=cdict['hparams'] 
	for (key, value) in cdict.items(): 
		if(type(value)!=dict):
			config[key]=cdict[key]
		if(key=='dp'):
			for (key2, value2) in value.items(): 
				config[key2]=value2
	config['uuid']=str(uuid.uuid4())[:5]
	config['root_path']=os.path.abspath(exp_root+"results_"+str(config['backbone'])+"_"+str(config['focal']))
	config['model_file']=os.path.abspath(config['root_path']+"/model_weights/model-"+str(config['backbone'])+"-weights_f"+str(config['focal'])+"_"+config['uuid']+".hdf5")
	config['predictions_file']=os.path.abspath(config['root_path']+"/submission.csv")
	return config
	

	
def fix_seeds(seedn):
	seed(seedn)
	tf.compat.v1.set_random_seed(seedn)
	random.seed(seedn)
	np.random.seed(seedn)
	os.environ['PYTHONHASHSEED']=str(seedn)


@ex.automain
def main(_run):
	fix_seeds(2)

	hparams=tune_config(_run.config)
	_run.config=hparams
	os.environ["CUDA_VISIBLE_DEVICES"] = hparams['GPUs'] 

	init_filestructure(hparams)
	
	# create model
	model = SCDModel(hparams)
	if(hparams['debug']):
		model.summary()
    
    # train model
	#train_data, val_data = loadData("train", hparams, val_split=hparams['val_split']  )
	train_data, val_data = loadSavedData("train", hparams )
 	#print('batch: '+str(hparams['batch_size'])+' lr: '+str(hparams['initial_learning_rate']))

	history=model.train(train_data, val_data, plot_results=False) 
	#print(history)
	#print(history.keys())
	val_metric=np.min(history['val_mean_absolute_error'])
	for k in history.keys(): 
		for i in range(len(history[k])): 
			_run.log_scalar(k,history[k][i],i) 

	#predictions=model.predict(val_data)
	#predictions = [y for x in predictions for y in x]
 
	val_y=([])
	predictions=([])
	for k in range(0,len(val_data)):
		batch = val_data.__getitem__(k)
		val_y=np.append(val_y,batch[1])

		predicts=model.predictVal(batch[0])
		predicts = [y for x in predicts for y in x]
		predictions=np.append(predictions,predicts)


 
	predictions_df = pd.DataFrame({"predictions": predictions, "targets": val_y})
	filename =hparams['root_path']+"/artifacts/predictions_df"+hparams['uuid']+".pickle"
	predictions_df.to_pickle(filename)
	_run.add_artifact(filename, name="predictions_df", content_type="application/octet-stream")

	std=repr(np.std(val_y-predictions, axis= 0))
	p99=repr(np.percentile(val_y-predictions, 99, axis=0))
	mae=repr(np.sum(np.absolute(val_y-predictions)))
	dist_y=np.array([(1/(1 + x/12.6572)) for x in val_y])
	dist_pred=np.array([(1/(1 + x/12.6572)) for x in predictions])
	error=np.absolute(dist_y-dist_pred)
	distortion=repr(np.sum(error))
	_run.log_scalar('std',std)
	_run.log_scalar('p99',p99)
	_run.log_scalar('mae',mae)
	_run.log_scalar('distortion',distortion)
	

	if(val_metric<2 and hparams['img_set']=='real'):
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
	return val_metric
	#test_data,_  = loadData("test",hparams, args.backgrounds,sub_sample_size=-1 )
	
	# counter=0
	# for batch in train_data:
	# 	counter+=1
	# 	print(counter,"-batch, size:",len(batch),len(batch[0]),len(batch[0][0]),len(batch[0][0][0]),len(batch[0][0][0][0]) )
		
	# 	###Batch = [3,batch_size,244,244,3] 
	# 	###Batch[0] = Image
	# 	# If we merge focals into input...
	# 	###Batch[1] = Focal 
	# 	###Batch[2] = Distance
	# 	# Else
	# 	###Batch[1] = Distance
	# 	print(batch[1])
	# 	for i in range (0,2):  
	# 	 	image = batch[0][i]
	# 	# 	print(batch[1][i],batch[2][i])
	# 	 	plt.imshow(image)
	# 	 	plt.show()
    #model.train(train_data, val_data, plot_results=True)

    # submit model
    #test_data, _ = loadData("test", hparams,args.backgrounds)
    #model.create_submit(test_data)
