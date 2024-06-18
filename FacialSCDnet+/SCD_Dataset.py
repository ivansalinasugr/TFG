from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import time

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, hparams, augment=False):
        self.images = images
        self.labels = labels
        self.dim = hparams['image_shape'][0]
        self.augment = augment
        self.config = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images.iloc[idx]['Path']
        img = Image.open(img_path).convert('RGB')

        if self.augment:
            img = self.augmentor(img)
        else:
            img = self.light_augmentor(img)
        
        label = self.labels[idx]

        return img, label

    def light_augmentor(self, image):
        padding = (0, 0, 0, 0)
        # Definimos la cantidad de relleno que necesitamos para alcanzar una dimensión de 768x768
        if (image.width != image.height):
            maximo = max(image.width, image.height)
            padding_left = max(0, (maximo - image.width) // 2)
            padding_right = max(0, (maximo - image.width + 1) // 2)  # Aseguramos que el relleno sea par
            padding_top = max(0, (maximo - image.height) // 2)
            padding_bottom = max(0, (maximo - image.height + 1) // 2)  # Aseguramos que el relleno sea par
            padding = (padding_left, padding_top, padding_right, padding_bottom)

        transform = transforms.Compose([
            transforms.Pad(padding),
            transforms.Resize((self.dim, self.dim)),
            transforms.ToTensor()
        ])

        return transform(image)
    
    # def augmentor(self, image):
    #     transform = transforms.Compose([
    #         transforms.Resize((self.dim, self.dim)),
    #         transforms.ToTensor(),
    #         transforms.RandomApply([
    #             transforms.RandomAffine(degrees=(-15, 15), translate=(0.2, 0.2))], 
    #             p=1),
    #         transforms.RandomApply([
    #             transforms.GaussianBlur(1.0)],
    #             p=0.25),
    #         transforms.RandomAdjustSharpness(2, p=0.25),
    #         transforms.RandomApply([
    #             transforms.ColorJitter(brightness=0.1, contrast=0.1,hue=0.1)],
    #             p=0.25),
    #         transforms.RandomApply([
    #             transforms.RandomErasing(scale=(0.02, 0.05), ratio=(0.5, 1.5), value=0.5)],
    #             p=0.25),
    #     ])

    #     return transform(image)

    def augmentor(self, image):
        transform = transforms.Compose([
            transforms.Resize((self.dim, self.dim)),
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=(-15, 15), translate=(0.2, 0.2))], 
                p=1),
            transforms.RandomApply([
                transforms.GaussianBlur(1.0)],
                p=1),
            transforms.RandomAdjustSharpness(2, p=1),
        ])

        return transform(image)