import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import random

from class_renaming import class_mapping

# ----------------------------------------------------------------------------#

AUGMENTATIONS = 2
ADD_AUGMENTATIONS = False

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

def dataAugmentation(image_path):
    """
    Augmentation function used to take each letter and create a new randomly
    augmented version of it through:
        → Random rotation (-5 to 5 degrees).
        → Random contrast (0.8 to 1.2).
    """
    try:
        original_img = Image.open(image_path).convert('L')
        
        for i in range(AUGMENTATIONS):
            ''' Random rotation (-5 to +5 degrees). '''
            angle = random.uniform(-5, 5)
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
            
            ''' Random contrast (0.8 to 1.2). '''
            contrast_factor = random.uniform(0.8, 1.2)
            contrast_enhancer = ImageEnhance.Contrast(rotated_img)
            contrast_img = contrast_enhancer.enhance(contrast_factor)
            
            # Saving the augmented image.
            base, ext = os.path.splitext(image_path)
            new_path = f"{base}_aug{i}{ext}"
            contrast_img.save(new_path)
            
    except Exception as e:
        print(f"Error augmenting image {image_path}: {str(e)}")


class GreekLetterDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=ADD_AUGMENTATIONS):
        self.root_dir = root_dir
        self.transform = transform
        self.images = self._load_images()
        self.augment = augment
        self.classes = sorted(list(set([img[1] for img in self.images]))) 
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        original_classes = sorted(list(set([img[1] for img in self.images])))
        self.classes = [class_mapping.get(cls, cls) for cls in original_classes]
        self.class_to_idx = {class_mapping.get(cls, cls): i for i, cls in enumerate(original_classes)}
        self.original_to_idx = {cls: i for i, cls in enumerate(original_classes)}
        
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
                        
                        # Skipping to avoid duplicates!
                        if '_aug' in img_name:
                            continue
                        
                        img_path = os.path.join(letter_path, img_name)
                        
                        if ADD_AUGMENTATIONS:
                            dataAugmentation(img_path) 
                            
                        images.append((img_path, class_name))
                        
                        # Including augmented versions in the dataset.
                        if ADD_AUGMENTATIONS:
                            base, ext = os.path.splitext(img_name)
                            for i in range(AUGMENTATIONS):
                                aug_path = os.path.join(letter_path, 
                                                        f"{base}_aug{i}{ext}")
                                if os.path.exists(aug_path):
                                    images.append((aug_path, class_name))
        
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.original_to_idx[label]        
        return image, label_idx
    
# ----------------------------------------------------------------------------#

