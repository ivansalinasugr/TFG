import os,gc
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import ssl
import torch
import mlflow
import metrics as met
from SCD_Dataset import CustomImageDataset
from SCD_Model import SCDModel
from sklearn.model_selection import train_test_split
from telegram import Bot
import asyncio
import json

# For telegram configuration
async def enviar_mensaje(token, chat_id, mensaje):
	bot = Bot(token=token)
	await bot.send_message(chat_id=chat_id, text=mensaje)

with open('telegram.json', 'r') as file:
    config = json.load(file)

TOKEN = config['TOKEN']
CHAT_ID = config['CHAT_ID']

ssl._create_default_https_context = ssl._create_unverified_context

# src_root="/mnt/homeGPU/isalinas/FSCDnet"

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Pruebas MLflow")

src_root="/Users/ivansalinas/GitHub/TFG/FSCDnet/SCDnet"

def init_filestructure(hparams):
	if not os.path.exists(hparams['root_path']):
		os.mkdir(hparams['root_path']) 
	if not os.path.exists(hparams['root_path'] + '/best_models'):
		os.mkdir(hparams['root_path']+'/best_models')
	if not os.path.exists(hparams['root_path'] + '/predictions'):
		os.mkdir(hparams['root_path']+'/predictions')

def base_conf(): 
	hparams = { 
		'focal':27, # 27 35 53 83.6
		#''' Train Parameters '''
		'n_epochs' : 10, # number of epochs for training
		'batch_size' :  32,	# size of the batch 
		'val_split' : 0.2 , # portion of the data that will be used for training
		#''' Network Parameters '''
		'patience' : 3, # learning rate will be reduced after this many epochs if the validation loss is not improving
		'early_stop' : 6, # training will be stopped after this many epochs without the validation loss improving
		'initial_learning_rate' : 0.0005, # initial learning rate
		'minimum_learning_rate' : 1e-12, # minimum value for the learning rate after reduction
		'learning_rate_drop' :  0.5,  # factor by which the learning rate will be reduced
		'with_dropout' : True, # include a dropout layer if overfitting
		'dropout':0.5, # amount of dropout
		'with_batchnormalization' : True, # include a BatchNormalization layer if overfitting
		'metrics': ['mae','mre'], # metrics
		'loss':'distortloss', # loss function
		'optimizer':'adam', # optimizers
		'size_fc2':4096, # input size of fc2 layer
		#''' General Data Parameters '''
		'name': "FacialNet", # name of the dataset
		'overwrite' : True,  # if True, will overwrite previous files. If False, will use previously written files.
		'backbone': 'vgg', # vgg, resnet, inception
		'image_shape' : (224, 224, 3), # this determines what shape the images will be cropped/resampled to
		'load_model' : False, # if True, use a model from files
		'load_weight_model': False, # if True, use weights from a model in files
		'debug': False # if True, shows arquitecture
	}
	hparams['root_path'] = src_root
	hparams['predictions_file'] = os.path.abspath(src_root + "/predictions/predictions_" + hparams['backbone'] + "_f" + str(hparams['focal']) + ".csv")
	hparams['model_file']=os.path.abspath(src_root + "/best_models/best_model_f"+str(hparams['focal'])+ ".pth")

	hparams['training_file'] = os.path.abspath(src_root + "/input/train_labels.csv")
	hparams['testing_file'] = os.path.abspath(src_root + "/input/test_labels.csv")

	return hparams

def purgeData(data,focal):
	if(focal==35):
		purged_data=data[data['Focal'].isin([32.1,33.9,33,35,35.4])]
	elif(focal==55):
		purged_data=data[data['Focal'].isin([52.1,53,53.5,54.4,55])]
	elif(focal==85):
		purged_data=data[data['Focal'].isin([83.6,84.1])]
	else:
		purged_data=data[data['Focal']==focal] 

	return purged_data

def loadData(db, hparams, val_split=0.2, sub_sample_size=-1):
	# Loads data
	if db == "train":
		data = pd.read_csv(hparams['training_file'],sep=',')
		data = purgeData(data, hparams['focal'])
		image_data=data.loc[:, data.columns != 'Distance'][:sub_sample_size]
		labels = data['Distance'].values[:sub_sample_size]
		if val_split > 0:
			X_train, X_test, Y_train, Y_test = train_test_split(image_data, labels, test_size=val_split, random_state=42)
			print("train images: "+str(len(X_train))+" val images: "+str(len(X_test)))
			train_dataset = CustomImageDataset(X_train, Y_train, hparams, augment=True)
			val_dataset = CustomImageDataset(X_test, Y_test, hparams, augment=False)
			train_data = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
			val_data = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)
			return train_data, val_data
		else:
			train_dataset = CustomImageDataset(image_data, labels, hparams, augment=True)
			train_data = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
			return train_data, None
	else:
		data = pd.read_csv(hparams['testing_file'],sep=',')
		data = purgeData(data,hparams['focal'])
		image_data =data.loc[:, data.columns != 'Distance'][:sub_sample_size]
		print("test images",len(image_data.index))
		labels = data['Distance'].values[:sub_sample_size]
		if sub_sample_size== -1 :
			test_dataset = CustomImageDataset(image_data, labels, hparams, augment=False)
			test_data = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)
			return test_data, None
		else:
			test_dataset = CustomImageDataset(image_data[:sub_sample_size], labels[:sub_sample_size], hparams, augment=False)
			test_data = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)
			return test_data, None
		
def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

def main(hparams):
	# Set seeds
	seed = 42
	set_seed(seed)
	
	init_filestructure(hparams)
	
	# Load models
	train_data, val_data = loadData('train', hparams)
	
	# Start an MLflow run
	with mlflow.start_run(run_name = "pruebas_mlflow"):
		mlflow.log_params(hparams)
		
		# Create model
		model = SCDModel(hparams)
		if(hparams['debug']):
			model.summary()
		
		# Train model
		train_losses, val_losses = model.train(train_data, val_data, plot_results=False)

		# Get best validation loss
		distortion = np.min(val_losses)

		# Log the model
		mlflow.pytorch.log_model(model.getModel(), "model")
		
		# Predict labels for test
		predicted_labels, true_labels = model.predict(val_data)
		predictions_df = pd.DataFrame({"predictions": predicted_labels, "targets": true_labels})
		predictions_df.to_csv(hparams['predictions_file'])
		mlflow.log_artifact(hparams['predictions_file'])
		
		# Calculate metrics
		results = met.calculate_metrics(hparams['metrics'], torch.tensor(true_labels), torch.tensor(predicted_labels))
		
		# Record the metrics in MLflow
		for metric_name in hparams['metrics']:
			mlflow.log_metrics(results[metric_name])
		
		# best_val_metric=np.min(val_metrics['MAE'])
	
	# if(best_val_metric<7):
	# 	print("Good Validation Result, we will test the model")
	# 	test_data, _ = loadData("test", hparams)
	# 	test_predicted_labels, test_true_labels = model.predict(test_data)
	# 	test_predictions_df = pd.DataFrame({"predictions": test_predicted_labels, "targets": test_true_labels})
	# 	test_predictions_df.to_csv("predictions_test.csv")
		
	# 	test_loss, test_mae, test_mape = model.evaluate(test_data)
	# 	print(f"Test loss: {test_loss}, test MAE: {test_mae}, test MRE: {test_mape} %")
	
	model.release()
	
	del model, train_data, val_data
	torch.cuda.empty_cache()
	gc.collect()
	
	return distortion

if __name__ == "__main__":
	hparams = base_conf()
	distortion = main(hparams)
	mensaje = f"El script ha finalizado. La distorsión es: {distortion}"
	asyncio.run(enviar_mensaje(TOKEN, CHAT_ID, mensaje))