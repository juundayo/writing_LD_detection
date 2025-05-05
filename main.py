# ----------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

import data_loading as dl

# ----------------------------------------------------------------------------#

EPOCHS = 1
BATCH_SIZE = 32

# ----------------------------------------------------------------------------#
''' Loading data through the data_loading.py file.'''
# Image transform.
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
    
# Loading the dataset and using an 80% train - 20% test split.
full_dataset = dl.GreekLetterDataset(root_dir='Thesis/GreekLetters', 
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
dysgraphia model. We use an attention mechanism to focus on the letters and the
lines themselves, opposed to the white spaces between them.
'''

# ----------------------------------------------------------------------------#

# Channel Attention.
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return torch.sigmoid(out).unsqueeze(2).unsqueeze(3)
    
# Spatial Attention.
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

# CBAM.
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ResidualBlock.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        
        if self.downsample:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out
    
# ----------------------------------------------------------------------------#

class LetterOCR(nn.Module):
    def __init__(self, num_classes):
        super(LetterOCR, self).__init__()
        
        # Initial convolution block.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks.
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Multi-scale feature fusion.
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', 
                               align_corners=True)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', 
                               align_corners=True)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1)
        
        # Final classification.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 + 256 + 128 + 64, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, 
                                    downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial features.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks.
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Multi-scale feature fusion.
        y1 = self.conv2(x4)
        y1 = self.up1(y1)
        
        y2 = self.conv3(y1 + x3)
        y2 = self.up2(y2)
        
        y3 = self.conv4(y2 + x2)
        
        # Global average pooling.
        x4_pool = self.avgpool(x4).view(x4.size(0), -1)
        y1_pool = self.avgpool(y1).view(y1.size(0), -1)
        y2_pool = self.avgpool(y2).view(y2.size(0), -1)
        y3_pool = self.avgpool(y3).view(y3.size(0), -1)
        
        # Concatenating features.
        features = torch.cat([x4_pool, y1_pool, y2_pool, y3_pool], dim=1)
        
        # Classification.
        features = self.dropout(features)
        out = self.fc(features)
        
        return out
    
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
    model = LetterOCR(num_classes=len(full_dataset.classes)).to(device)
    
    # Loss function with label smoothing.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay.
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                     patience=3, factor=0.5)
    
    # Early stopping!
    best_acc = 0.0
    patience = 5
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
            #torch.save(model.state_dict(), "ocr_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                pbar.close()
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    pbar.close()
    return model

# ----------------------------------------------------------------------------#

model = train_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_acc = evaluate_model(model, test_loader, device)
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
