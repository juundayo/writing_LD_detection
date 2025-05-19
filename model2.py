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

import data_loading as dl

# ----------------------------------------------------------------------------#

EPOCHS = 1000
PATIENCE = 15
BATCH_SIZE = 16
TRAIN = True
SAVE_PATH = "/home/ml3/Desktop/Thesis/MODELOTYPA2.pth"
LOAD_PATH = "/home/ml3/Desktop/Thesis/MODELOTYPA.pth"

# ----------------------------------------------------------------------------#

'''
Model creation for character recognition.

The model is trained to recognise each letter separately. Each image that it 
is trained on has letters between the lines which will later help with the 
writing disorder model. We use an attention mechanism to focus on the letters 
and the lines themselves, opposed to the white spaces between them.
'''

# ----------------------------------------------------------------------------#

class OCR(nn.Module):
    def __init__(self, num_classes):
        super(OCR, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

        self.fc1 = nn.Linear(32*6*6, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x
    
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
        transforms.Resize((32, 32)),
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

