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
import metrics2 as met
import seaborn as sns
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

def load_config(file_path, args):
	with open(file_path, 'r') as f:
		config = json.load(f)

	config['root_path'] = args.src_root

	config['training_file'] = os.path.abspath(args.src_root + "/input/train_labels.csv")
	config['testing_file'] = os.path.abspath(args.src_root + "/input/test_labels.csv")
	config['load_model'] = True

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
	

def crear_predictions_df(hparams, device, args):
	#focals = [27, 35, 53, 83.6]
	focals = [35]
	dfs = {}
	for focal in focals:
		hparams["focal"] = focal
		hparams['model_file']=os.path.abspath(args.src_root + "/best_models/best_model_f"+str(hparams['focal'])+ ".pth")
		# Load models
		_, val_data = loadData('train', hparams)

		# Create model
		model = SCDModel(hparams, device)
		
		predicted_labels, true_labels = model.predict(val_data)
		predicted_labels_numpy = [tensor.item() for tensor in predicted_labels]
		true_labels_numpy = [tensor.item() for tensor in true_labels]
		pred_error = [true -pred for pred, true in zip(predicted_labels_numpy, true_labels_numpy)]

		predictions_df = pd.DataFrame({"predictions": predicted_labels_numpy, "targets": true_labels_numpy, "pred_error": pred_error})
		predictions_df.to_csv("validation.csv")
		dfs[focal] = predictions_df
		model.release()
		del model, val_data
	return dfs

def main(hparams, args):
	# Set seeds
	seed = 42
	set_seed(seed)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	dfs = crear_predictions_df(hparams, device, args)
	#focals = [27, 35, 53, 83.6]
	focals = [35]

	metrics = ['mae', 'mre', 'distortloss']
	for focal in focals:
		val_metrics = met.calculate_metrics(metrics, torch.tensor(dfs[focal]['targets']), torch.tensor(dfs[focal]['predictions']))
		print(f'Focal: {focal}')
		for metric_name, metric_result in val_metrics.items():
			print(metric_name)
			print(round(metric_result[f'{metric_name}_mean'], 3))
			print(round(metric_result[f'{metric_name}_std'], 3))
			print(round(metric_result[f'{metric_name}_median'], 3))
			print(round(metric_result[f'{metric_name}_min'], 3))
			print(round(metric_result[f'{metric_name}_perc_90'], 3))
			print(round(metric_result[f'{metric_name}_perc_95'], 3))
			print(round(metric_result[f'{metric_name}_perc_99'], 3))

	# Configurar el número de subgráficas y su disposición
	num_subplots = len(focals)
	cols = len(focals)  # Cambia según la cantidad de columnas que desees
	rows = 1  # Redondeo hacia arriba de la división entera

	# Configurar el tamaño de la figura
	plt.figure(figsize=(5*num_subplots, 5))  # Ajusta el tamaño según tus preferencias

	# Título general para las cuatro figuras
	plt.suptitle('Comparación de etiquetas predichas vs objetivos', fontsize=4*num_subplots)

	# Iterar sobre cada focal y generar la gráfica correspondiente
	for i, focal in enumerate(focals, 1):
		plt.subplot(rows, cols, i)
		sns.regplot(data=dfs[focal], y="predictions", x="targets", marker="+")
		plt.title(f'Focal {focal}')
		plt.xlabel('Objetivos')
		plt.ylabel('Predicciones')

	# Ajustar el espacio entre subgráficas
	plt.tight_layout()

	# Guardar la figura con todas las gráficas juntas en un solo archivo
	plt.savefig('SCDnet/graficas/comparacion_etiquetas.png')
	plt.close()

	# Configurar el número de subgráficas y su disposición
	num_subplots = len(focals)
	cols = len(focals)  # 4 boxplots y 4 violinplots
	rows = 2

	# Configurar el tamaño de la figura
	plt.figure(figsize=(5*num_subplots, 10))  # Ajusta el tamaño según tus preferencias

	# Título general para las cuatro figuras
	plt.suptitle('Distribución de errores de predicción', fontsize=16)

	# Generar los boxplots para cada focal en la primera fila
	for i, focal in enumerate(focals, 1):
		plt.subplot(rows, cols, i)
		sns.boxplot(data=dfs[focal],y='pred_error', orient='v')
		plt.title(f'Focal {focal}')
		plt.ylabel('Error de predicción')

	# Generar los violinplots para cada focal en la segunda fila
	for i, focal in enumerate(focals, num_subplots+1):  # Comienza desde el índice 5
		plt.subplot(rows, cols, i)
		sns.violinplot(data=dfs[focal],y='pred_error', orient='v')
		plt.title(f'Focal {focal}')
		plt.ylabel('Error de predicción')

	# Ajustar el espacio entre subgráficas
	plt.tight_layout()

	# Guardar la figura con todas las gráficas juntas en un solo archivo
	plt.savefig('SCDnet/graficas/comparacion_box_violin.png')
	plt.close()
	
	torch.cuda.empty_cache()
	gc.collect()
	
	return 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="FacialSCDnet programa")
	
	# We define arguments for our program
	parser.add_argument("--src_root", type=str, required=True, help="Directorio raíz de origen")
	#parser.add_argument("--run_name", type=str, required=True, help="Nombre del run en MLflow")
	
	args = parser.parse_args()

	hparams = load_config('/mnt/homeGPU/isalinas/FSCDnet/SCDnet/codigos/config.json', args)
	with open('/mnt/homeGPU/isalinas/FSCDnet/SCDnet/codigos/telegram.json', 'r') as file:
		config = json.load(file)
	# # Set our tracking server uri for logging
	# mlflow.set_tracking_uri("file:///mnt/homeGPU/isalinas/FSCDnet/SCDnet/mlruns")

	# mlflow.set_experiment("Conjunto métricas")

	distortion = main(hparams, args)
