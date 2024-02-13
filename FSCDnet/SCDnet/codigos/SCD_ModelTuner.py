import torch
from torchvision import models
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import losses as lo
import gc
import mlflow
import numpy as np
import torch

class EarlyStopping:
	def __init__(self, patience, path, delta=0, verbose=False):
		self.patience = patience # number of epochs with no improvement after which training will be stopped
		self.delta = delta # minimum change to qualify as an improvement
		self.verbose = verbose # if True, prints messages.
		self.path = path # file to save the model checkpoint
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf

	def __call__(self, val_loss, model):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
		elif score < self.best_score + self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		# Save model if validation loss improve
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model, self.path)
		self.val_loss_min = val_loss

class SCDModel():
	def __init__(self, hparams):
		self.config = hparams
		#self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = "mps" if torch.backends.mps.is_available() else "cpu"

		if self.config['load_model']:
			self.model = torch.load(self.config['model_file'])
		else:
			self.model = self.create_model()
			if self.config['load_weight_model']:
				self.load_weights_only()

	def summary(self):
		print(self.model)

	def getModel(self):
		return self.model
		
	def release(self):
		del self.model
		gc.collect()
		torch.cuda.empty_cache()

	def create_model(self):
		if self.config['backbone'] == 'vgg':
			backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
			num_features = backbone.classifier[0].in_features
			# Congelar todas las capas excepto la última capa classifier
			for param in backbone.parameters():
				param.requires_grad = False
		
			layers = [
				nn.Linear(num_features, 4096),
				nn.ReLU(inplace=True),
			]
			if(self.config['with_dropout']):
				layers.append(nn.Dropout(self.config['dropout']))

			layers.append(nn.Linear(4096, self.config['size_fc2']))
			layers.append(nn.ReLU(inplace=True))
			if(self.config['with_dropout']):
				layers.append(nn.Dropout(self.config['dropout']))
			if(self.config['with_batchnormalization']):
				layers.append(nn.BatchNorm1d(4096))
			layers.append(nn.Linear(self.config['size_fc2'], 1))

			backbone.classifier = nn.Sequential(*layers)

		# Por implementar
		# elif self.config['backbone'] == 'resnet':
		#     backbone = models.resnet50(weigths=models.ResNet50_Weights.DEFAULT)
		#     num_features = backbone.fc.in_features
		#     backbone.fc = nn.Linear(num_features, 1)
		# elif self.config['backbone'] == 'inception':
		#     backbone = models.inception_v3(weigths=models.Inception_V3_Weights.DEFAULT)
		#     num_features = backbone.fc.in_features
		#     backbone.fc = nn.Linear(num_features, 1)
		
		func_loss = self.config['loss']
		
		if(func_loss == 'distortloss'):
			self.loss = lo.distortloss 
		elif(func_loss == 'distortmae'):
			self.loss = lo.distortmae
		
		if(self.config['optimizer'] == 'adam'):
			self.optimizer = optim.Adam(backbone.parameters(), lr=self.config['initial_learning_rate'])
		else:
			self.optimizer = optim.SGD(backbone.parameters(), lr=self.config['initial_learning_rate'])

		return backbone
	
	def forward(self, x):
		return self.model(x)


	def train(self, train_loader, val_loader, plot_results=False):
		print("Starting training")
		# Reduce learning rate if no improvement is seen
		scheduler = ReduceLROnPlateau(self.optimizer,
									  mode='min',
									  patience=self.config['patience'],
									  factor=self.config['learning_rate_drop'],
									  min_lr=self.config['minimum_learning_rate'])

		# Early stopping
		early_stopping = EarlyStopping(patience=self.config['early_stop'], path=self.config['model_file'], verbose=False)

		val_losses = []

		self.model.to(self.device)

		# Train the model
		for epoch in range(self.config['n_epochs']):
			self.model.train()
			train_loss = 0.0
			for data, target in train_loader:
				data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
				self.optimizer.zero_grad()
				output = self.model(data)
				losses = self.loss(output.squeeze(), target)

				train_loss += losses.item()
				losses.backward()
				self.optimizer.step()
			
			train_loss /= len(train_loader)
			
			self.model.eval()
			val_loss = 0.0
			with torch.no_grad():
				for data, target in val_loader:
					data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
					output = self.model(data)
					val_losses_batch = self.loss(output.squeeze(), target)

					val_loss += val_losses_batch.item()

			val_loss /= len(val_loader)

			scheduler.step(val_loss)
			
			val_losses.append(val_loss)

			# Log the loss
			mlflow.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch+1)

			# Check early stopping
			early_stopping(val_loss, self.model)
			
			if early_stopping.early_stop:
				print("Early stopping")
				break
		
		self.load_weights_only() # Load best weights
		
		print("Finished training")

		return val_losses

	def load_weights_only(self):
		# Load the saved model file
		model = torch.load(self.config['model_file'])
		
		# Load the weights into the model
		self.model.load_state_dict(model.state_dict())

	def predict(self, test_data, load_weights=False):
		if load_weights:
			# Load pretrained weights if load_weights is True
			self.load_weights_only()
		
		self.model.eval()
		predictions = []
		targets = []
		# Make prediction
		with torch.no_grad():
			for data, target in test_data:
				data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
				output = self.model(data)
				predictions.extend(output.squeeze().cpu().numpy())
				targets.extend(target.cpu().numpy())
				
		
		return predictions, targets