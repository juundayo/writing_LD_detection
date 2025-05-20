# ----------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import time
import random

import data_loading as dl

# ----------------------------------------------------------------------------#

EPOCHS = 3000
PATIENCE = 3000
BATCH_SIZE = 16
TRAIN = True
SAVE_PATH = "/home/ml3/Desktop/Thesis/rs1000_tt2_v1.pth"
LOAD_PATH = "/home/ml3/Desktop/Thesis/rs10_tt2_v2.pth"

# ----------------------------------------------------------------------------#

'''
Model creation for character recognition.

The model is trained to recognise each letter separately. Each image that it 
is trained on has letters between the lines which will later help with the 
writing disorder model. We use an attention mechanism to focus on the letters 
and the lines themselves, opposed to the white spaces between them.
'''

# ----------------------------------------------------------------------------#

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze.
        y = self.avg_pool(x).view(b, c)

        # Excitation.
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
# ----------------------------------------------------------------------------#

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, use_se=False):
        super().__init__()

        self.use_se = use_se

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, 
                          bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None
        if use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)
    
# ----------------------------------------------------------------------------#

class OCR(nn.Module):
    def __init__(self, num_classes):
        super(OCR, self).__init__()
        
        # Initial convolution.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, 
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # Residual stages.
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, 
                                       use_se=True)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, 
                                       use_se=True)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, 
                                       use_se=True)

        # Self-attention on the feature map.
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

        # Final convolution.
        self.layer4 = self._make_layer(256,512, blocks=2, stride=2, use_se=True)

        # Classification.
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, blocks, stride, use_se):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride=stride,
                                 use_se=use_se))
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, stride=1, 
                                     use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        # [B,1,45,80] -> conv/bn/relu
        x = self.relu(self.bn1(self.conv1(x)))  # -> [B,64, ~23,40]
        x = self.layer1(x)                      # -> [B,64,23,40]
        x = self.layer2(x)                      # -> [B,128,11,20]
        x = self.layer3(x)                      # -> [B,256, 6,10]

        batch, neurons, h, w = x.size()
        x_flat = x.view(batch, neurons, h*w).permute(2, 0, 1) # -> [h*w, B, C]
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(1,2,0).view(batch, neurons, h, w)

        x = x + attn_out                        # Residual connection.
        x = self.layer4(x)                      # -> [B,512, 3, 5]
        x = self.global_pool(x).view(batch, -1) # -> [B,512]
        x = self.fc(x)                          # -> [B,num_classes]
        return x
    
# ----------------------------------------------------------------------------#

'''
Using more augmentations for better training and test performance.
We have already applied random rotations and random contrast to 2.500
images, and now we take these images and add more small augmentations 
with a 50% probability. Specifically, we use:
    → Random affine.
    → Random perspective.
    → Gaussian blur.
    → Color jitter.
'''
class DynamicAugmentations:
    def __init__(self):
        self.augmentations = [
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), 
                                    scale=(0.95, 1.05,), shear=5),
            transforms.RandomPerspective(0.3, p=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(contrast=(0.8,1.2)),
                transforms.Lambda(lambda x: x + torch.randn_like(x)*0.03)
            ], p=0.3)
        ]

    def __call__(self, img):
        # 50% chance per augmentation.
        for aug in self.augmentations:
            if random.random() < 0.5:  
                img = aug(img)
        return img

# ----------------------------------------------------------------------------#

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# ----------------------------------------------------------------------------#

def train_model():
    # Initializing the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCR(num_classes=len(full_dataset.classes)).to(device)

    labels = [label for _, label in train_dataset]
    class_counts = Counter(labels)
    weights = 1. / torch.tensor([class_counts[i] for i in range(len(full_dataset.classes))], 
                                dtype=torch.float)

    # Loss function with label smoothing.
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), 
                                    label_smoothing=0.1)
    
    # Optimizer with weight decay.
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                     patience=3, 
                                                     min_lr=1e-4,
                                                     factor=0.2)
    
    # Early stopping!
    best_acc = 0.0
    patience = PATIENCE
    patience_counter = 0
        
    pbar = tqdm(range(EPOCHS), desc="Training Progress", unit="epoch", 
                position=0, leave=True)
    
    # Training!
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Adding dynamic augmentations to images (0.5 chance).
            aug = DynamicAugmentations()
            images = torch.stack([aug(img) for img in images])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        test_acc = evaluate_model(model, test_loader, device)
        epoch_time = time.time() - start_time
        
        # Progress bar update.
        pbar.set_postfix({
            'Loss': f"{running_loss/len(train_loader):.4f}",
            'Train Acc': f"{train_acc:.2f}%",
            'Test Acc': f"{test_acc:.2f}%", 
            'Time': f"{epoch_time:.2f}s",
            'Best': f"{best_acc:.2f}%"
        })
        
        # Scheduler update.
        scheduler.step(test_acc)
        
        # Early stopping check.
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                pbar.close()
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    pbar.close()
    return model

# ----------------------------------------------------------------------------#

def plot_confusion_matrix(model, test_loader, device, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------#

if __name__ == "__main__":

    ''' Loading data through the data_loading.py file.'''
    # Image transform.
    data_transforms = transforms.Compose([
        transforms.Resize((512, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
        
    # Loading the dataset and using an 80% train - 20% test split.
    full_dataset = dl.GreekLetterDataset(root_dir='/home/ml3/Desktop/Thesis/.venv/Data/GreekLetters', 
                                    transform=data_transforms)

    train_dataset, test_dataset = full_dataset.get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Found {len(full_dataset.classes)} classes: {full_dataset.classes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if TRAIN:
        model = train_model()
        test_acc = evaluate_model(model, test_loader, device)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    else:
        model = OCR(num_classes=len(full_dataset.classes)).to(device)
        model.load_state_dict(torch.load(LOAD_PATH))
        model.eval()
        print("Loaded the model successfully.")

    # Displaying a confusion matrix.
    plot_confusion_matrix(model, test_loader, device, full_dataset.classes)

# ----------------------------------------------------------------------------#

