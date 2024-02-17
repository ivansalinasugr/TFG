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
import optuna
from optuna.samplers import TPESampler
import json
import argparse

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
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("Pruebas MLflow")

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

	config['training_file'] = os.path.abspath(args.src_root + "/input/train_labels.csv")
	config['testing_file'] = os.path.abspath(args.src_root + "/input/test_labels.csv")

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
	seed = 1
	set_seed(seed)
	
	init_filestructure(hparams)
	print(hparams['predictions_file'])
	
	# Load models
	train_data, val_data = loadData('train', hparams)

	# Create model
	model = SCDModel(hparams, compute_metrics=False, verbose=False)
	if(hparams['debug']):
		model.summary()
	
	# Train model
	_ , val_losses = model.train(train_data, val_data)

	distortion = np.min(val_losses)

	model.release()
	
	del model, train_data, val_data
	torch.cuda.empty_cache()
	gc.collect()
	
	return distortion

def objective(trial, args):
	# Parameters to optimize
	hparams = load_config('config.json', args)
	hparams['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64]) 
	hparams['initial_learning_rate'] = trial.suggest_float('initial_learning_rate', 1e-5, 1e-2, log=True)
	hparams['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
	hparams['size_fc2'] = trial.suggest_categorical('size_fc2', [1024, 2048, 4096])
	with mlflow.start_run():
		# Set run name
		mlflow.set_tag("mlflow.runName", "Trial_"+str(trial.number))
		# Report the hyperparameters to MLflow
		mlflow.log_params(trial.params)
		
		distortion = main(hparams)
		
	return distortion  # Metric we want to optimize

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="FacialSCDnet programa")
	
	# Definimos los argumentos que nuestro programa aceptará
	parser.add_argument("--src_root", type=str, required=True, help="Directorio raíz de origen")
	parser.add_argument("--focal", type=float, required=True, help="Valor de focal")
	parser.add_argument("--n_epochs", type=int, required=True, help="Número de épocas")
	
	# Parseamos los argumentos de la línea de comandos
	args = parser.parse_args()

	study = optuna.create_study(study_name='Tuneo2', direction='minimize', sampler=TPESampler(seed=1)) 
	study.optimize(lambda trial: objective(trial, args), n_trials=2) 

	best_trial = study.best_trial
	print("Best trial: {}\n".format(best_trial.number))
	print(" - Params: ")
	for key, value in best_trial.params.items():
		print(" {}: {}".format(key, value))
	
	print(" - Value: {}\n".format(best_trial.value))

	mensaje = "El script de tuneo de hiperparámetros ha finalizado"
	asyncio.run(enviar_mensaje(TOKEN, CHAT_ID, mensaje))