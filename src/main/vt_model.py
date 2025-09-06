# ----------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import time
import random
import cv2
from PIL import Image
import os 

import data_loading as dl

# ----------------------------------------------------------------------------#

SAVE_PATH = "/home/ml3/Desktop/Thesis/Models/SD_OCR6.pth"
LOAD_PATH = "/home/ml3/Desktop/Thesis/Models/SD_OCR6.pth"
DATA_DIR = '/home/ml3/Desktop/Thesis/.venv/Data/GreekLetters'
EPOCHS = 2000
PATIENCE = 150
BATCH_SIZE = 16
IMG_HEIGHT = 512
IMG_WIDTH = 78
TRAIN = False

SIMILAR_PAIRS = {
    'Ε': 'ε', 'ε': 'Ε',
    'Θ': 'θ', 'θ': 'Θ',
    'Ι': 'ι', 'ι': 'Ι',
    'Κ': 'κ', 'κ': 'Κ',
    'Ο': 'ο', 'ο': 'Ο',
    'Π': 'π', 'π': 'Π',
    'Ρ': 'ρ', 'ρ': 'Ρ',
    'Τ': 'τ', 'τ': 'Τ',
    'Χ': 'χ', 'χ': 'Χ',
    'Ψ': 'ψ', 'ψ': 'Ψ'
}

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
        self.dropout = nn.Dropout(p=0.4)
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
        x = self.dropout(x)
        x = self.fc(x)                          # -> [B,num_classes]
        return x
    
# ----------------------------------------------------------------------------#

'''
Using more augmentations for better training and test performance.
We have already applied random rotations and random contrast to 1.500
images, and now we take these images and add more small augmentations 
with a 55% probability. Specifically, we use:
    → Random affine.
    → Random perspective.
    → Gaussian blur.
    → Color jitter.
'''
def replicate_pad(img_tensor, pad=20):
    """
    Converts tensor to OpenCV image, adds a 
    pad with replicate and returns a tensor.
    """
    device = img_tensor.device
    img_tensor_cpu = img_tensor.cpu()
    img = transforms.ToPILImage()(img_tensor_cpu)
    img_np = np.array(img)

    padded = cv2.copyMakeBorder(img_np, pad, pad, pad, pad, 
                                cv2.BORDER_REPLICATE)
    padded_img = Image.fromarray(padded)

    return transforms.ToTensor()(padded_img).to(device)

class DynamicAugmentations:
    def __init__(self, device='cuda'):
        self.device = device
        self.augmentations = [
            ("RandomAffine", transforms.RandomAffine(
                degrees=7, translate=(0.05, 0.08), scale=(0.95, 1.05), shear=6,
                interpolation=transforms.InterpolationMode.BILINEAR
            )),
            ("RandomPerspective", transforms.RandomPerspective(
                distortion_scale=0.3, p=0.4,
                interpolation=transforms.InterpolationMode.BILINEAR
            )),
            ("GaussianBlur", transforms.GaussianBlur(kernel_size=3, 
                                                     sigma=(0.1, 1.0))),
            ("ColorJitter+Noise", transforms.Compose([
                transforms.ColorJitter(contrast=(0.5, 1.9)),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03)
            ]))
        ]

    def __call__(self, img_tensor):
        # 55% chance per augmentation.
        for name, aug in self.augmentations:
            if random.random() < 0.55:
                if name in ["RandomAffine", "RandomPerspective"]:
                    padded = replicate_pad(img_tensor, pad=20)
                    augmented = aug(padded)
                    # Cropping back to the original size.
                    _, h, w = img_tensor.shape
                    _, H, W = augmented.shape
                    top = (H - h) // 2
                    left = (W - w) // 2
                    augmented = augmented[:, top:top+h, left:left+w]
                    img_tensor = augmented
                else:
                    img_tensor = aug(img_tensor).to(self.device)

        return img_tensor

# ----------------------------------------------------------------------------#

'''
Using a custom loss function that combines the base cross-entropy loss
with a similarity penalty for characters that are visually similar.
This loss function will help the model to not only classify characters
correctly but also to recognize similar characters as correct predictions.
'''
class SimilarCharacterLoss(nn.Module):
    def __init__(self, num_classes, class_to_idx, weight=None, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.class_to_idx = class_to_idx
        self.base_loss = nn.CrossEntropyLoss(weight=weight, 
                                             label_smoothing=label_smoothing)
        
        # Creating the similarity matrix (0.5 penalty for similar characters).
        self.similarity_matrix = torch.eye(num_classes)
        
        # Fill in the similarity matrix
        for char1, char2 in SIMILAR_PAIRS.items():
            if char1 in class_to_idx and char2 in class_to_idx:
                idx1 = class_to_idx[char1]
                idx2 = class_to_idx[char2]
                self.similarity_matrix[idx1, idx2] = 0.7
                self.similarity_matrix[idx2, idx1] = 0.7

    def forward(self, inputs, targets):
        # Calculating base cross-entropy loss.
        base_loss = self.base_loss(inputs, targets)
        
        # Calculating similarity-adjusted loss.
        probs = F.softmax(inputs, dim=1)
        batch_size = targets.size(0)
        
        # Converting targets to one-hot labels.
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Applying the similarity matrix to targets.
        adjusted_targets = torch.matmul(targets_onehot, 
                                        self.similarity_matrix.to(inputs.device))
        
        # Calculating KL divergence between predictions and adjusted targets.
        similarity_loss = F.kl_div(probs.log(), adjusted_targets, 
                                   reduction='batchmean')
        
        # Combining losses.
        total_loss = base_loss + 0.2 * similarity_loss
        
        return total_loss
        
# ----------------------------------------------------------------------------#

def similar_character_accuracy(outputs, labels, classes):
    """
    Calculating both strict and lenient accuracy.
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    
    # Checking for similar character matches.
    similar_correct = 0
    for i in range(len(predicted)):
        pred_idx = predicted[i].item()
        true_idx = labels[i].item()
        pred_char = classes[pred_idx]
        true_char = classes[true_idx]
        if pred_char != true_char and pred_char in SIMILAR_PAIRS and SIMILAR_PAIRS[pred_char] == true_char:
            similar_correct += 1
    
    total = labels.size(0)
    strict_acc = 100 * correct / total
    lenient_acc = 100 * (correct + 0.8 * similar_correct) / total
    
    return strict_acc, lenient_acc

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

    # Class weights for imbalanced data.
    labels = [label for _, label in train_dataset]
    class_counts = Counter(labels)

    # Class weights = inverse frequency.
    class_weights = 1. / torch.tensor(
        [class_counts[i] for i in range(len(full_dataset.classes))], 
        dtype=torch.float
    )
    # per-sample weights
    sample_weights = torch.tensor([class_weights[label] for label in labels],
                                  dtype=torch.float)

    # sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,     
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Custom loss function with similar character handling.
    criterion = SimilarCharacterLoss(
        num_classes=len(full_dataset.classes),
        class_to_idx=full_dataset.class_to_idx,
        label_smoothing=0.3
    )
    
    # Optimizer and scheduler.
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, min_lr=1e-4, factor=0.2
    )
    
    # Early stopping and tracking.
    best_acc = 0.0
    patience = PATIENCE
    patience_counter = 0
    
    # Metric tracking for plotting.
    train_loss_history = []
    train_strict_acc_history = []
    train_lenient_acc_history = []
    test_strict_acc_history = []
    test_lenient_acc_history = []
        
    pbar = tqdm(range(EPOCHS), desc="Training Progress", unit="epoch", position=0, leave=True)
    
    # トレーニング！
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_similar = 0
        epoch_total = 0
        start_time = time.time()
        
        for images, labels in train_loader:
            # Moving data to device.
            images, labels = images.to(device), labels.to(device)
            
            # Applying augmentations.
            if TRAIN:  
                aug = DynamicAugmentations(device=device)
                images = torch.stack([aug(img) for img in images])

            # Forward pass!
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass!
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Calculating batch metrics.
            batch_size = labels.size(0)
            epoch_total += batch_size
            epoch_loss += loss.item() * batch_size  # Weighted by batch size
            
            # Getting predictions.
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            epoch_correct += batch_correct
            
            # Tracking similar characters (only for incorrect predictions).
            incorrect_mask = (predicted != labels)
            incorrect_preds = predicted[incorrect_mask]
            incorrect_labels = labels[incorrect_mask]
            
            for pred, true in zip(incorrect_preds, incorrect_labels):
                pred_char = full_dataset.classes[pred.item()]
                true_char = full_dataset.classes[true.item()]
                if pred_char in SIMILAR_PAIRS and SIMILAR_PAIRS[pred_char] == true_char:
                    epoch_similar += 1
        
        # Calculating epoch metrics.
        train_loss = epoch_loss / epoch_total
        train_strict_acc = 100 * epoch_correct / epoch_total
        train_lenient_acc = 100 * (epoch_correct + 0.75 * epoch_similar) / epoch_total
        
        # Test evaluation
        model.eval()
        test_correct = 0
        test_similar = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                batch_size = labels.size(0)
                test_total += batch_size
                test_correct += (predicted == labels).sum().item()
                
                # Track similar characters in test set
                incorrect_mask = (predicted != labels)
                incorrect_preds = predicted[incorrect_mask]
                incorrect_labels = labels[incorrect_mask]
                
                for pred, true in zip(incorrect_preds, incorrect_labels):
                    pred_char = full_dataset.classes[pred.item()]
                    true_char = full_dataset.classes[true.item()]
                    if pred_char in SIMILAR_PAIRS and SIMILAR_PAIRS[pred_char] == true_char:
                        test_similar += 1
        
        test_strict_acc = 100 * test_correct / test_total
        test_lenient_acc = 100 * (test_correct + 0.5 * test_similar) / test_total
        
        # Store metrics
        train_loss_history.append(train_loss)
        train_strict_acc_history.append(train_strict_acc)
        train_lenient_acc_history.append(train_lenient_acc)
        test_strict_acc_history.append(test_strict_acc)
        test_lenient_acc_history.append(test_lenient_acc)
        
        # Update progress bar
        epoch_time = time.time() - start_time
        pbar.set_postfix({
            'Loss': f"{train_loss:.4f}",
            'Train Acc (S/L)': f"{train_strict_acc:.2f}%/{train_lenient_acc:.2f}%",
            'Test Acc (S/L)': f"{test_strict_acc:.2f}%/{test_lenient_acc:.2f}%", 
            'Time': f"{epoch_time:.2f}s",
            'Best': f"{best_acc:.2f}%"
        })
        
        # Learning rate scheduling
        scheduler.step(test_strict_acc)
        
        # Early stopping
        if test_strict_acc > best_acc:
            best_acc = test_strict_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                pbar.close()
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Final cleanup.
    pbar.close()
    
    # Plotting training curves.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_strict_acc_history, label='Train Strict Acc')
    plt.plot(train_lenient_acc_history, label='Train Lenient Acc')
    plt.plot(test_strict_acc_history, label='Test Strict Acc')
    plt.plot(test_lenient_acc_history, label='Test Lenient Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
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
    plt.savefig("confusion_matrix.png", bbox_inches='tight', pad_inches=0)

# ----------------------------------------------------------------------------#

def imshow(img_tensor):
    """Converts normalized image tensor to numpy image and denormalizes."""
    img = img_tensor.cpu().numpy().squeeze()
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    return img

def plot_predictions(model, test_loader, class_names, device, num_figures=10, img_per_fig=20, save_dir="output_debug_images"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    total_images = num_figures * img_per_fig
    shown = 0
    fig = None
    axes = None

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if shown >= total_images:
                    if fig is not None:
                        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                        fig.savefig(os.path.join(save_dir, f"predictions_batch_{fig_idx+1}.png"))
                        plt.close(fig)
                    return

                fig_idx = shown // img_per_fig
                subplot_idx = shown % img_per_fig

                if subplot_idx == 0:
                    if fig is not None:
                        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                        fig.savefig(os.path.join(save_dir, f"predictions_batch_{fig_idx}.png"))
                        plt.close(fig)
                    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
                    axes = axes.flatten()
                    fig.suptitle(f'Predictions: Batch {fig_idx + 1}', fontsize=16)

                ax = axes[subplot_idx]
                ax.imshow(imshow(images[i]), cmap='gray')
                ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", fontsize=10)
                ax.axis('off')

                shown += 1

        if fig is not None:
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(os.path.join(save_dir, f"predictions_batch_{fig_idx+1}.png"))
            plt.close(fig)

# ----------------------------------------------------------------------------#

def print_model_accuracy(model, test_loader, device, class_names):
    """
    Calculating and printing the final 
    model accuracy with detailed metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    similar_correct = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Check for similar character matches
            incorrect_mask = (predicted != labels)
            incorrect_preds = predicted[incorrect_mask]
            incorrect_labels = labels[incorrect_mask]
            
            for pred, true in zip(incorrect_preds, incorrect_labels):
                pred_char = class_names[pred.item()]
                true_char = class_names[true.item()]
                if pred_char in SIMILAR_PAIRS and SIMILAR_PAIRS[pred_char] == true_char:
                    similar_correct += 1
    
    # Calculating accuracies.
    strict_acc = 100 * correct / total
    lenient_acc = 100 * (correct + 1 * similar_correct) / total
    
    # Printing results.
    print("\n" + "="*60)
    print("MODEL ACCURACY REPORT")
    print("="*60)
    print(f"Total test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Similar character matches: {similar_correct}")
    print(f"Strict Accuracy: {strict_acc:.2f}%")
    print(f"Lenient Accuracy: {lenient_acc:.2f}%")
    print("="*60)
    
    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracy = 100 * cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-class accuracy:")
    for i, cls in enumerate(class_names):
        print(f"{cls}: {class_accuracy[i]:.2f}%")
    
    return strict_acc, lenient_acc

# ----------------------------------------------------------------------------#

def plot_detailed_confusion_matrix(model, test_loader, device, class_names):
    """
    Updated confusion matrix with percentages and counts.
    """
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
    
    # Calculating percentages.
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Accuracy Percentage'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage with Count)')
    plt.tight_layout()
    plt.show()
    plt.savefig("detailed_confusion_matrix.png", bbox_inches='tight', pad_inches=0)

# ----------------------------------------------------------------------------#

if __name__ == "__main__":

    ''' Loading data through the data_loading.py file.'''
    # Image transform.
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
        
    # Loading the dataset and using an 80% train - 20% test split.
    full_dataset = dl.GreekLetterDataset(root_dir=DATA_DIR, 
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

    plot_confusion_matrix(model, test_loader, device, full_dataset.classes)
    plot_predictions(model, test_loader, full_dataset.classes, device)

    strict_acc, lenient_acc = print_model_accuracy(model, test_loader, device, full_dataset.classes)
    
    plot_detailed_confusion_matrix(model, test_loader, device, full_dataset.classes)
