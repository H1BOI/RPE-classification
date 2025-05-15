
import os
from skimage import io
from torch.utils.data import Dataset

# Dataset class
class RPEDataset(Dataset):
    def __init__(self, dir="Good RPE Crops", transform=None):
        self.dir = dir
        self.transform = transform
        self.samples = []

        # Map folder names to labels
        group_labels = {'young': 0, 'middle': 1, 'old': 2}

        # Load images from grouped folders
        for group_name, label in group_labels.items():
            folder_path = os.path.join(self.dir, group_name)
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.startswith('._'):
                        continue
                    img_path = os.path.join(folder_path, img_name)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return (image, label)
