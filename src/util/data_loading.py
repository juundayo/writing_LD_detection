# ----------------------------------------------------------------------------#

import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from class_renaming import class_mapping

# ---------------------------------------------------------------------------- #

AUGMENTATIONS = 1500
RANDOM_STATE = 101
AUGMENT = True

# ---------------------------------------------------------------------------- #

'''
Data loading and preparation.

File layout:
    currentfolder/Data/GreekLetters/

    → Inside GreekLetters can be found two folders named CAPS and SMALL.
        → Both folders contain two separate folders named SingleCharacters and
          DoubleCharacters.
          → Inside each one of the folders can be found seperate folders, each
            containing a single letter or a combination of two letters - 
            depending on the folder.
'''

# ----------------------------------------------------------------------------#

def dataAugmentation(image_path, aug_number):
    """
    Augmentation function used to take each letter and create a new randomly
    augmented version of it through:
        → Random rotation (-6 to 6).
        → Random contrast (1 to 1.6).
    """
    try:
        original_img = Image.open(image_path).convert('L')

        # Dynamic upper y-axis extension. 
        base_name = os.path.basename(image_path)
        if base_name.endswith(".tif") and "_" in base_name:
            original_img = extend_upper_y_axis(original_img)

        for i in range(aug_number):
            ''' Random rotation (-6 to +6 degrees). '''
            angle = random.uniform(-6, 6)
            img_array = np.array(original_img)
            
            # Image center.
            height, width = img_array.shape[:2] 
            image_center = (width/2, height/2)
            
            # Rotation matrix.
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            
            # Rotating.
            rotated_array = cv2.warpAffine(img_array, rot_mat, 
                                         (width, height),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            
            # Converting back to PIL Image!
            rotated_img = Image.fromarray(rotated_array)
            
            ''' Random contrast (1.0 to 1.6). '''
            contrast_factor = random.uniform(1.0, 1.6)
            contrast_enhancer = ImageEnhance.Contrast(rotated_img)
            contrast_img = contrast_enhancer.enhance(contrast_factor)
            
            # Saving the augmented image.
            base, ext = os.path.splitext(image_path)
            new_path = f"{base}_aug{i}{ext}"
            contrast_img.save(new_path)
            
    except Exception as e:
        print(f"Error augmenting image {image_path}: {str(e)}")

# ----------------------------------------------------------------------------#

def extend_upper_y_axis(img: Image.Image) -> Image.Image:
    """
    Extends the upper y-axis of the image with white background.
    Random extension up to 2.5x original height.
    """
    width, height = img.size
    
    # Maximum new height (up to 2.5x original).
    max_extra_height = int(1.5 * height) 
    extra_height = random.randint(0, max_extra_height)
    
    # If no extension, return original
    if extra_height == 0:
        return img
    
    new_height = height + extra_height
    
    # Create white background
    new_img = Image.new("L", (width, new_height), color=255)
    
    # Paste original image at (0, extra_height)
    new_img.paste(img, (0, extra_height))
    
    return new_img

# ----------------------------------------------------------------------------#

class GreekLetterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train_dataset = None
        self.test_dataset = None
        self.load_images()
        
        # Getting all unique classes!
        all_classes = sorted(list(set([img[1] for img in self.train_dataset + self.test_dataset])))
        
        self.classes = []
        for cls in all_classes:
            mapped_cls = class_mapping.get(cls, cls)
            self.classes.append(mapped_cls)
                
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.original_to_idx = {orig_cls: i for i, orig_cls in enumerate(all_classes)}
        
    def iterate_through(self):
        images_by_class = {}
        character_types = ['SMALL', 'CAPS']
        sub_folders = ['SingleCharacters']

        for char_type in character_types:
            char_type_dir = os.path.join(self.root_dir, char_type)
            if not os.path.exists(char_type_dir):
                continue
                
            for sub_folder in sub_folders:
                sub_folder_dir = os.path.join(char_type_dir, sub_folder)
                if not os.path.exists(sub_folder_dir):
                    continue
                
                for letter_folder in sorted(os.listdir(sub_folder_dir)):
                    letter_path = os.path.join(sub_folder_dir, letter_folder)
                    if not os.path.isdir(letter_path):
                        continue
                        
                    # Saving the letter as a class name.
                    class_name = letter_folder
                    images_by_class.setdefault(class_name, [])
                    
                    # Skipping to avoid duplicates!
                    for img_name in os.listdir(letter_path):
                        if '_aug' in img_name:
                            continue

                        img_path = os.path.join(letter_path, img_name)
                        images_by_class[class_name].append(img_path)

        return images_by_class

    def load_images(self):
        all_images = self.iterate_through()

        train_dataset = []
        test_dataset = []

        for class_name, image_paths in all_images.items():
            train_imgs, test_imgs = train_test_split(
                image_paths, test_size=0.2, random_state=RANDOM_STATE
            )

            train_dataset.extend([(path, class_name) for path in train_imgs])
            test_dataset.extend([(path, class_name) for path in test_imgs])
            
            '''Creating augmentations for the train and test dataset.'''
            if AUGMENT:
                print("Creating augmentations...")
                # Augmentations in the train set.
                print(f"Creating augmentations for the train set for class: {class_name}")
                for path in train_imgs:
                    for i in range(AUGMENTATIONS):
                        dataAugmentation(path, 1)
                        base, ext = os.path.splitext(path)
                        aug_path = f"{base}_aug0{ext}"

                        if os.path.exists(aug_path):
                            renamed_path = f"{base}_augT{i}{ext}"
                            os.rename(aug_path, renamed_path)
                            train_dataset.append((renamed_path, class_name))

                # Augmentations in the test set.
                for path in test_imgs:
                    for i in range(AUGMENTATIONS):
                        dataAugmentation(path, 1)
                        base, ext = os.path.splitext(path)
                        aug_path = f"{base}_aug0{ext}"

                        if os.path.exists(aug_path):
                            renamed_path = f"{base}_augV{i}{ext}"
                            os.rename(aug_path, renamed_path)
                            test_dataset.append((renamed_path, class_name))

        if AUGMENT:
            print(f"Created augmentations successfully!")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("Loaded datasets successfully!")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.original_to_idx[label]        
        return image, label_idx
    
    def get_datasets(self):
        return (
            GreekLetterSubDataset(self.train_dataset, self.transform,
                                   self.original_to_idx),
            GreekLetterSubDataset(self.test_dataset, self.transform, 
                                  self.original_to_idx)
        )

# ----------------------------------------------------------------------------#

class GreekLetterSubDataset(Dataset):
    def __init__(self, samples, transform, original_to_idx):
        self.samples = samples
        self.transform = transform
        self.original_to_idx = original_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)
        label_idx = self.original_to_idx[label]

        return image, label_idx
    
# ----------------------------------------------------------------------------#

def plot_augmentation_samples(root_dir, num_folders=3, num_augmentations=3):
    """
    Plot original images and their respective
    augmentations from random folders.
    """
    character_folders = []
    
    # Searching through CAPS and SMALL folders.
    for char_type in ['CAPS', 'SMALL']:
        char_type_dir = os.path.join(root_dir, char_type, 'SingleCharacters')
        if not os.path.exists(char_type_dir):
            continue
            
        for letter_folder in os.listdir(char_type_dir):
            letter_path = os.path.join(char_type_dir, letter_folder)
            if os.path.isdir(letter_path):
                character_folders.append(letter_path)
    
    # Randomly selecting folders.
    selected_folders = random.sample(character_folders, min(num_folders, len(character_folders)))
    
    fig, axes = plt.subplots(num_folders, 4, figsize=(15, 5*num_folders))
    if num_folders == 1:
        axes = [axes] 
    
    for i, folder in enumerate(selected_folders):
        all_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', 'tif'))]
        
        # Finding original images (those without '_aug' in name).
        original_images = [f for f in all_files if '_aug' not in f]
        
        if not original_images:
            print(f"No original images found in {folder}")
            continue
            
        # Selecting a random original image.
        original_img_name = random.choice(original_images)
        original_img_path = os.path.join(folder, original_img_name)
        
        # Finding augmentations for the selected image.
        base_name, ext = os.path.splitext(original_img_name)
        augmentations = [f for f in all_files if f.startswith(base_name) and '_aug' in f]
        
        # Selecting random augmentations.
        selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
        
        # Loading the images.
        original_img = Image.open(original_img_path)
        aug_imgs = [Image.open(os.path.join(folder, aug)) for aug in selected_augmentations]
        
        # Plotting the original in top left.
        axes[i][0].imshow(original_img, cmap='gray')
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')
        
        for j, aug_img in enumerate(aug_imgs, start=1):
            axes[i][j].imshow(aug_img, cmap='gray')
            axes[i][j].set_title(f'Augmentation {j}')
            axes[i][j].axis('off')
        
        for j in range(len(aug_imgs)+1, 4):
            axes[i][j].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png')
    plt.show()
    print("Saved plot as 'augmentation_samples.png'")

# ----------------------------------------------------------------------------#

if __name__ == '__main__':
    root_directory = '/home/ml3/Desktop/Thesis/.venv/Data/GreekLetters' 
    plot_augmentation_samples(root_directory)
