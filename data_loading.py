import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import random
from sklearn.model_selection import train_test_split

from class_renaming import class_mapping

# ----------------------------------------------------------------------------#

AUGMENTATIONS = 3
SPACE_AUGMENTATIONS = 20

# ----------------------------------------------------------------------------#

'''
Data loading and preparation.

File layout:
    currentfolder/Data/GreekLetters/

    → Inside GreekLetters can be found two folders named CAPS and SMALL.
        →　Both folders contain two separate folders named SingleCharacters and
          DoubleCharacters.
          →　Inside each one of the folders can be found seperate folders, each
            containing a single letter or a combination of two letters - 
            depending on the folder.
'''

# ----------------------------------------------------------------------------#

def dataAugmentation(image_path, aug_number):
    """
    Augmentation function used to take each letter and create a new randomly
    augmented version of it through:
        → Random rotation (-10 to 10).
        → Random contrast (1 to 1.5).
    """
    try:
        original_img = Image.open(image_path).convert('L')
        
        for i in range(aug_number):
            ''' Random rotation (-10 to +10 degrees). '''
            angle = random.uniform(-10, 10)
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
            
            ''' Random contrast (1.0 to 1.5). '''
            contrast_factor = random.uniform(1.0, 1.5)
            contrast_enhancer = ImageEnhance.Contrast(rotated_img)
            contrast_img = contrast_enhancer.enhance(contrast_factor)
            
            # Saving the augmented image.
            base, ext = os.path.splitext(image_path)
            new_path = f"{base}_aug{i}{ext}"
            contrast_img.save(new_path)
            
    except Exception as e:
        print(f"Error augmenting image {image_path}: {str(e)}")


class GreekLetterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train_dataset = None
        self.test_dataset = None
        self._load_images()
        
        # Getting all unique classes!
        all_classes = sorted(list(set([img[1] for img in self.train_dataset + self.test_dataset])))
        
        self.classes = []
        for cls in all_classes:
            mapped_cls = class_mapping.get(cls, cls)
            self.classes.append(mapped_cls)
                
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.original_to_idx = {orig_cls: i for i, orig_cls in enumerate(all_classes)}
        

    def _iterate_through(self, dataset=None, ADD_AUGMENTATIONS=False):
        images = []
        character_types = ['SMALL', 'SPECIAL']
        sub_folders = ['SingleCharacters', 'Symbols']

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

                    # Changing the amount of augmentations based on the class.
                    if class_name == ' ':
                        aug_number = SPACE_AUGMENTATIONS
                    else:
                        aug_number = AUGMENTATIONS

                    for img_name in os.listdir(letter_path):
                            
                        # Skipping to avoid duplicates!
                        if '_aug' in img_name:
                            continue
                    
                        img_path = os.path.join(letter_path, img_name)
                        
                        if ADD_AUGMENTATIONS:
                            dataAugmentation(img_path, aug_number) 

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

    def _load_images(self):

        full_dataset = self._iterate_through()

        train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.3, 
                                               random_state=1)

        augmented_dataset = self._iterate_through(train_dataset, 
                                                  ADD_AUGMENTATIONS=True)
        
        self.train_dataset = augmented_dataset
        self.test_dataset = test_dataset
    
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
            GreekLetterSubDataset(self.train_dataset, self.transform, self.original_to_idx),
            GreekLetterSubDataset(self.test_dataset, self.transform, self.original_to_idx)
        )
    
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

