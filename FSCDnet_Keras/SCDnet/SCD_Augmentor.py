import cv2
import numpy as np
from tensorflow.keras.utils import Sequence 
#from sklearn.preprocessing import MinMaxScaler
#ImgAug
import imgaug as ia
from imgaug import augmenters as iaa

class DataGenerator(Sequence):
	def __init__(self, image_data, labels, hparams, batch_size=64, shuffle=False, augment=False ):
		self.labels       = labels              # array of labels
		self.image_data   = image_data
		self.config		  = hparams
		self.dim          = self.config['image_shape']    # image dimensions
		self.batch_size   = batch_size          # batch size
		self.shuffle      = shuffle             # shuffle bool
		self.augment      = augment             # augment data bool
		self.images_paths,self.focals  = self.parse_data()
		self.on_epoch_end()
		ia.seed(self.config['imgaug_seed'])

	def parse_data(self):
		images_paths = self.image_data['Path'].values
		focals = self.image_data['Focal'].values
		return images_paths, focals

	def __len__(self):
		#'Denotes the number of batches per epoch'
		return int(np.floor(len(self.images_paths) / self.batch_size))

	def on_epoch_end(self):
		#'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.images_paths))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		#'Generate one batch of data'
		indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]    # selects indices of data for next batch
		# select data and load images
		label = np.array([self.labels[k] for k in indexes])	
		
		images = [cv2.imread(self.images_paths[k], 1) for k in indexes]

		if self.augment == True:        # preprocess and augment data
			images_aug = self.augmentor(images)
		else:
			images_aug	= self.light_augmentor(images)

		return np.array(images_aug), np.array(label)


	def light_augmentor(self, images):
		always = lambda aug: iaa.Sometimes(1, aug)
		seq = iaa.Sequential([
			always(iaa.PadToFixedSize(width=self.config['padToFixedW'], height=self.config['padToFixedH'], position="center")),
			always(iaa.Resize(self.dim[0]))
			])
		return seq(images=images)
 

	def augmentor(self, images):
		always = lambda aug: iaa.Sometimes(1, aug)
		seq = iaa.Sequential([
			always(iaa.PadToFixedSize(width=self.config['padToFixedW'], height=self.config['padToFixedH'], position="center")),
			always(iaa.Affine(
			 		translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
			 		rotate=(-15, 15), # rotate by -45 to +45 degrees
			 		order=1, # use nearest neighbour or bilinear interpolation (fast)
			 		cval=0, # if mode is constant, use a cval between 0 and 255
			 		mode="constant" )),
			always(iaa.Resize(self.dim[0])),
			iaa.SomeOf((0, 2),[
				iaa.OneOf([
					iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
					iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 7
					iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
					]),
				iaa.OneOf([
					iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)),
					iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.25)), # emboss images
					]),
				iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
				iaa.OneOf([
						iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
						iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
						iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False),
						iaa.Cutout(fill_mode="constant", cval=255)
					]),
				iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
				iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
				],random_order=True)
			])
		return seq(images=images)
