from torchvision import datasets
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class CustomImageFolder(Dataset):
    def __init__(self, root, meta_file, transform=None):
        self.root = root
        with open(meta_file, 'r') as f:
            self.image_paths = [os.path.join(root, line.strip()) for line in f.readlines()]
        self.transform = transform
        self.image_folder = datasets.ImageFolder(root)
        
        # Filter out indices of images to be included based on the meta file
        image_paths_set = set(self.image_paths)
        self.indices = [
            idx for idx, path in tqdm(enumerate(self.image_folder.samples))
            if os.path.join(self.root, path[0]) in image_paths_set
        ]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.image_folder[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
