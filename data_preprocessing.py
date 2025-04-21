import os
from skimage import io
from torch.utils.data import Dataset

# Dataset class
class RPEDataset(Dataset):
    # You can get the data from specific days based on what you put
    def __init__(self, days, dir="Good RPE Crops", transform=None):
        self.dir = dir
        self.days = days
        self.transform = transform
        self.samples = []

        # Get the days for file pathing
        age_labels = ['30_day_crops', '45_day_crops', '60_day_crops', '180_day_crops', '330_day_crops', '720_day_crops']

        # Create corresponding labels
        label_to_idx = {label: idx for idx, label in enumerate(age_labels)}

        # Grab the folder corresponding to the days you put
        for label in age_labels:
            if str(days) in label:
                folder = label
                break
        
        # Extract each image within that photo as an array and put them in a list
        for image in os.listdir(os.path.join(dir, folder)):
            image_path = os.path.join(dir, folder, image)
            self.samples.append((image_path, label_to_idx[folder]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return (image, label)