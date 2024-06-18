import os,gc,random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import ssl
import torch
import mlflow
import mlflow.pytorch
from SCD_Dataset import CustomImageDataset
from SCD_Model import SCDModel
from sklearn.model_selection import train_test_split
from telegram import Bot
import asyncio
import json
import argparse
import time
from numpy.random import seed
import metrics as met

# For telegram configuration
async def enviar_mensaje(token, chat_id, mensaje):
	bot = Bot(token=token)
	await bot.send_message(chat_id=chat_id, text=mensaje)

ssl._create_default_https_context = ssl._create_unverified_context

def init_filestructure(hparams):
	if not os.path.exists(hparams['root_path']):
		os.mkdir(hparams['root_path']) 
	if not os.path.exists(hparams['root_path'] + '/best_models'):
		os.mkdir(hparams['root_path']+'/best_models')
	if not os.path.exists(hparams['root_path'] + '/predictions'):
		os.mkdir(hparams['root_path']+'/predictions')

def load_config(file_path, args):
	with open(file_path, 'r') as f:
		config = json.load(f)

	config['focal'] = args.focal
	config['n_epochs'] = args.n_epochs
	config['root_path'] = args.src_root
	config['predictions_file'] = os.path.abspath(args.src_root + "/predictions/predictions_" + config['backbone'] + "_f" + str(config['focal']) + ".csv")
	config['model_file']=os.path.abspath(args.src_root + "/best_models/best_model_f"+str(config['focal'])+ ".pth")

	config['training_file'] = os.path.abspath(args.src_root + "/input/images_v3/train_labels.csv")
	config['testing_file'] = os.path.abspath(args.src_root + "/input/images_v3/test_labels.csv")

	return config

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
	kwargs = {'num_workers': 2, 'pin_memory': True}
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
			train_data = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, **kwargs)
			val_data = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=True, **kwargs)
			return train_data, val_data
		else:
			train_dataset = CustomImageDataset(image_data, labels, hparams, augment=True)
			train_data = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, **kwargs)
			return train_data, None
	else:
		data = pd.read_csv(hparams['testing_file'],sep=',')
		data = purgeData(data,hparams['focal'])
		image_data =data.loc[:, data.columns != 'Distance'][:sub_sample_size]
		print("test images",len(image_data.index))
		labels = data['Distance'].values[:sub_sample_size]
		if sub_sample_size== -1 :
			test_dataset = CustomImageDataset(image_data, labels, hparams, augment=False)
			test_data = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False, **kwargs)
			return test_data, None
		else:
			test_dataset = CustomImageDataset(image_data[:sub_sample_size], labels[:sub_sample_size], hparams, augment=False)
			test_data = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False, **kwargs)
			return test_data, None
		
def set_seed(sem):
	np.random.seed(sem)
	torch.manual_seed(sem)
	seed(sem)
	random.seed(sem)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(sem)
		torch.cuda.manual_seed_all(sem)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True

def main(hparams, args):
	# Set seeds
	seed = 42
	set_seed(seed)
	
	init_filestructure(hparams)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Load models
	train_data, val_data = loadData('train', hparams)
	
	# Start an MLflow run
	with mlflow.start_run(run_name = args.run_name):
		mlflow.log_params(hparams)

		# Create model
		model = SCDModel(hparams, device)
		if(hparams['debug']):
			model.summary()
		
		if(not hparams["evaluate_mode"]):
			train_losses, val_losses = model.train(train_data, val_data)

			# Get best validation loss
			distortion = np.min(val_losses)

			# Predict labels for test
			predicted_labels, true_labels = model.predict(val_data)
			predicted_labels_numpy = [tensor.item() for tensor in predicted_labels]
			true_labels_numpy = [tensor.item() for tensor in true_labels]

			predictions_df = pd.DataFrame({"predictions": predicted_labels_numpy, "targets": true_labels_numpy})
			predictions_df.to_csv(hparams['predictions_file'])
			mlflow.log_artifact(hparams['predictions_file'])
		else:
			predicted_labels, true_labels = model.predict(val_data)
			predicted_labels_numpy = [tensor.item() for tensor in predicted_labels]
			true_labels_numpy = [tensor.item() for tensor in true_labels]

			predictions_df = pd.DataFrame({"predictions": predicted_labels_numpy, "targets": true_labels_numpy})
			predictions_df.to_csv("validation.csv")
			metrics = ['r2', 'mae', 'mre', 'distortloss']

			val_metrics = met.calculate_metrics(metrics, torch.tensor(true_labels), torch.tensor(predicted_labels))
			for metric_name in metrics:
				print(f'val_{metric_name}: {val_metrics[metric_name][f"{metric_name}_value"]}')
			distortion = val_metrics["distortloss"]["distortloss_value"]
	
		model.release()
	
	del model, train_data, val_data
	torch.cuda.empty_cache()
	gc.collect()
	
	return distortion

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="FacialSCDnet programa")
	
	# We define arguments for our program
	parser.add_argument("--src_root", type=str, required=True, help="Directorio raíz de origen")
	parser.add_argument("--run_name", type=str, required=True, help="Nombre del run en MLflow")
	parser.add_argument("--focal", type=float, required=True, help="Valor de focal")
	parser.add_argument("--n_epochs", type=int, required=True, help="Número de épocas")
	
	args = parser.parse_args()

	hparams = load_config('/mnt/homeGPU/isalinas/FSCDnet/SCDnet/codigos/config.json', args)
	with open('/mnt/homeGPU/isalinas/FSCDnet/SCDnet/codigos/telegram.json', 'r') as file:
		config = json.load(file)
	# Set our tracking server uri for logging
	mlflow.set_tracking_uri("file:///mnt/homeGPU/isalinas/FSCDnet/SCDnet/mlruns")

	# Create a new MLflow Experiment
	if(hparams["modo_prueba"]):
		mlflow.set_experiment("MLflow pruebas")
	else:
		mlflow.set_experiment("Conjunto nuevo")

	TOKEN = config['TOKEN']
	CHAT_ID = config['CHAT_ID']

	mode = "evaluate" if hparams["evaluate_mode"] else "training"
	message = "Start " + mode + " F" + str(hparams["focal"])
	asyncio.run(enviar_mensaje(TOKEN, CHAT_ID, message))

	# Save the start time
	start = time.time()

	distortion = main(hparams, args)

	# Save the end time
	end = time.time()

	# Calculate the time difference
	total_time = end - start

	# Calculate hours, minutes, and seconds
	hours = int(total_time // 3600)
	minutes = int((total_time % 3600) // 60)
	seconds = int(total_time % 60)

	# Prepare message with results
	message = "The script F_" +str(hparams["focal"]) + f" has finished. The distortion is: {distortion}, {mode} took {hours} hours, {minutes} minutes, and {seconds} seconds."

	asyncio.run(enviar_mensaje(TOKEN, CHAT_ID, message))
