from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, hparams, augment=False):
        self.images = images
        self.labels = labels
        self.dim = hparams['image_shape'][0]
        self.augment = augment
        self.config		  = hparams

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
        transform = transforms.Compose([
            transforms.Resize((self.dim, self.dim)),
            transforms.ToTensor()
        ])

        return transform(image)
    
    def augmentor(self, image):
        transform = transforms.Compose([
            transforms.Resize((self.dim, self.dim)),
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2))], 
                p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(1.0)],
                p=0.25),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1, contrast=0.1,)],
                p=0.25),
            transforms.RandomApply([
                transforms.RandomErasing(scale=(0.02, 0.05), ratio=(0.1, 0.5), value=0)],
                p=0.15),
            transforms.RandomApply([
                transforms.Grayscale(3)],
                p=0.15)
        ])

        return transform(image)