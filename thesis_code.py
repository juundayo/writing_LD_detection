# ----------------------------------------------------------------------------#

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------------------------#

BATCH_SIZE = 32

# ----------------------------------------------------------------------------#

'''
Data loading and preparation.

File layout:
    currentfolder/Thesis/GreekLetters/

    → Inside GreekLetters can be found two folders named CAPS and SMALL.
        →　Both folders contain two separate folders named SingleCharacters and
          DoubleCharacters.
          →　Inside each one of the folders can be found seperate folders, each
            containing a single letter or a combination of two letters - 
            depending on the folder.
'''

# ----------------------------------------------------------------------------#

class GreekLetterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = self._load_images()
        self.classes = sorted(list(set([img[1] for img in self.images]))) 
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def _load_images(self):
        images = []
        character_types = ['CAPS', 'SMALL']
        sub_folders = ['DoubleCharacters', 'SingleCharacters']
        
        for char_type in character_types:
            char_type_dir = os.path.join(self.root_dir, char_type)
            if not os.path.exists(char_type_dir):
                print('TEST - Error in Caps-Small character type!')
                continue
                
            for sub_folder in sub_folders:
                sub_folder_dir = os.path.join(char_type_dir, sub_folder)
                if not os.path.exists(sub_folder_dir):
                    print('TEST - Error in Double-Single character type!')
                    continue
                
                for letter_folder in sorted(os.listdir(sub_folder_dir)):
                    letter_path = os.path.join(sub_folder_dir, letter_folder)
                    if not os.path.isdir(letter_path):
                        print('TEST - Error in letters!')
                        continue
                        
                    # Saving the letter as a class name.
                    class_name = letter_folder
                    
                    for img_name in os.listdir(letter_path):
                        img_path = os.path.join(letter_path, img_name)
                        images.append((img_path, class_name))
        
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('L') # Grayscale conversion.
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.class_to_idx[label] # Label to index conversion.
        
        return image, label_idx
    
# ----------------------------------------------------------------------------#

# Image transform.
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
    
# Loading the dataset. Using an 80% train - 20% test split.
full_dataset = GreekLetterDataset(root_dir='Thesis/GreekLetters', 
                                  transform=data_transforms)
train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, 
                                               random_state=1)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Found {len(full_dataset.classes)} classes: {full_dataset.classes}")

# ----------------------------------------------------------------------------#

'''
Model creation for character recognition.

The model is trained to recognise each letter separately. Each image that it 
is trained on has lines between the letters which will later help with the 
dysgraphia model.
'''

# ----------------------------------------------------------------------------#

class GreekLetterCNN(nn.Module):
    def __init__(self, num_classes):
        super(GreekLetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# ----------------------------------------------------------------------------#



    