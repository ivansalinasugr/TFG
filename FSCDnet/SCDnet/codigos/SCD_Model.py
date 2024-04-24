import torch
from torchvision import models
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from losses import DistortLoss
import gc
import mlflow
import metrics as met
import numpy as np
import torch
import time

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
		elif score <= self.best_score + self.delta:
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
		torch.save(model.state_dict(), self.path)
		self.val_loss_min = val_loss

class SCDModel():
	def __init__(self, hparams, device, compute_metrics = True, verbose = True):
		self.config = hparams
		self.device = device
		if(self.device == 'cuda'):
			print("Using GPU")
		else:
			print("Using CPU")
			
		self.compute_metrics = compute_metrics
		self.verbose = verbose

		self.model = self.create_model()
		if self.config['load_model']:
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
			# Freeze all layers
			for param in backbone.parameters():
				param.requires_grad = False
		
			# Create new layers
			layers = [
				nn.Linear(num_features, 4096),
				nn.ReLU(inplace=True),
			]
			if(self.config['with_dropout']):
				layers.append(nn.Dropout(self.config['dropout']))

			layers.append(nn.Linear(4096, self.config['size_fc2']))
			layers.append(nn.ReLU(inplace=True))
			if(self.config['with_batchnormalization']):
				layers.append(nn.BatchNorm1d(self.config['size_fc2']))
			if(self.config['with_dropout']):
				layers.append(nn.Dropout(self.config['dropout']))
			layers.append(nn.Linear(self.config['size_fc2'], 1))

			# Set new layers as classifier
			backbone.classifier = nn.Sequential(*layers)
		
		backbone = nn.DataParallel(backbone)
		
		backbone.to(self.device)

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
			self.loss = DistortLoss()
		
		if(self.config['optimizer'] == 'adam'):
			self.optimizer = optim.Adam(backbone.parameters(), lr=self.config['initial_learning_rate'])
		else:
			self.optimizer = optim.SGD(backbone.parameters(), lr=self.config['initial_learning_rate'])

		return backbone
	
	def forward(self, x):
		return self.model(x)

	def train(self, train_loader, val_loader, plot_results=False):
		print("Starting training")
		focal=self.config['focal']
		# Reduce learning rate if no improvement is seen
		scheduler = ReduceLROnPlateau(self.optimizer,
									  mode='min',
									  patience=self.config['patience'],
									  factor=self.config['learning_rate_drop'],
									  min_lr=self.config['minimum_learning_rate'])

		# Early stopping
		early_stopping = EarlyStopping(patience=self.config['early_stop'], path=f'checkpoint_F{focal}.pth', verbose=self.verbose)

		train_losses = []
		val_losses = []
		
		# Train the model
		for epoch in range(self.config['n_epochs']):
			self.model.train()
			train_loss = torch.tensor(0.0, device=self.device)
			train_predictions = []
			train_targets = []
			for data, target in train_loader:
				data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
				self.optimizer.zero_grad()
				output = self.model(data)
				losses = self.loss(output.squeeze(), target)
				train_predictions.extend(output)
				train_targets.extend(target)

				train_loss += losses
				losses.backward()
				self.optimizer.step()
			
			train_loss /= len(train_loader.dataset)
			train_loss = train_loss.item()
			
			self.model.eval()
			val_loss = torch.tensor(0.0, device=self.device)
			val_predictions = []
			val_targets = []
			with torch.no_grad():
				for data, target in val_loader:
					data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
					output = self.model(data)
					val_losses_batch = self.loss(output.squeeze(), target)
					val_predictions.extend(output)
					val_targets.extend(target)

					val_loss += val_losses_batch

			val_loss /= len(val_loader.dataset)
			val_loss = val_loss.item()

			scheduler.step(val_loss)
			
			train_losses.append(train_loss)
			val_losses.append(val_loss)

			# Calcular métricas y registrar en MLflow
			if self.compute_metrics:
				train_metrics = met.calculate_metrics(self.config['metrics'], torch.tensor(train_targets), torch.tensor(train_predictions))
				val_metrics = met.calculate_metrics(self.config['metrics'], torch.tensor(val_targets), torch.tensor(val_predictions))

				# Log the metrics
				for metric_name in self.config['metrics']:
					mlflow.log_metric(f'train_{metric_name}', train_metrics[metric_name][f'{metric_name}_value'], step=epoch+1)
					mlflow.log_metric(f'val_{metric_name}', val_metrics[metric_name][f'{metric_name}_value'], step=epoch+1)

			# Log the loss
			mlflow.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch+1)
			
			if self.verbose:
				print(f'Epoch {epoch + 1}/{self.config["n_epochs"]}, Train Loss: {train_loss}, Val Loss: {val_loss}')

			# Check early stopping
			early_stopping(val_loss, self.model)
			
			if early_stopping.early_stop:
				if self.verbose:
					print("Early stopping")
				break
			
			torch.cuda.empty_cache()

		# if plot_results:
		#     fig, ax = plt.subplots(3, 1, figsize=(6, 12))
		#     ax[0].plot(train_losses, label="TrainLoss")
		#     ax[0].plot(val_losses, label="ValLoss")
		#     ax[0].legend(loc='best', shadow=True)
		#     ax[1].plot(train_metrics['MAE'], label="TrainMAE")
		#     ax[1].plot(val_metrics['MAE'], label="ValMAE")
		#     ax[1].legend(loc='best', shadow=True)
		#     ax[2].plot(train_metrics['MAPE'], label="TrainMAPE")
		#     ax[2].plot(val_metrics['MAPE'], label="ValMAPE")
		#     ax[2].legend(loc='best', shadow=True)
		#     plt.show()
		

		# Load best weights encounter in training
		self.model.load_state_dict(torch.load(f'checkpoint_F{focal}.pth'))

		# Save model
		torch.save(self.model.state_dict(), self.config['model_file'])

		print("Finished training")

		return train_losses, val_losses

	def load_weights_only(self):
		# Load the weights into the model
		self.model.load_state_dict(torch.load(self.config['model_file']))

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
				predictions.extend(output)
				targets.extend(target)
		
		return predictions, targets

		