# ----------------------------------------------------------------------------#

import os
import numpy as np
import torch
import cv2
import time
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from skimage.transform import (hough_line, hough_line_peaks)
from PIL import Image, ImageOps
from skimage import io
from collections import defaultdict

from vt_model import OCR
from data_loading import class_mapping

# ----------------------------------------------------------------------------#

MODEL_PATH = "/home/ml3/Desktop/Thesis/Models/rs1_tt2_v2.pth"

IMG_PATH = "/home/ml3/Desktop/Thesis/two_mimir.jpg"
#IMG_PATH = "/home/ml3/Downloads/input3.jpg"

DUMP = "/home/ml3/Desktop/Thesis/LetterDump"

OUTPUT_FOLDER = "/home/ml3/Desktop/Thesis/LetterCrops"
BLOCKS_FOLDER = "/home/ml3/Desktop/Thesis/BlockImages"

IMG_HEIGHT = 512
IMG_WIDTH = 78

SEARCH_TEST = False

# ----------------------------------------------------------------------------#

class GreekTextRecognizer:
    def __init__(self, device):
        os.makedirs(BLOCKS_FOLDER, exist_ok=True)
        for filename in os.listdir(BLOCKS_FOLDER):
            file_path = os.path.join(BLOCKS_FOLDER, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        
        
        # Dictionary that stores the coordinates of each letter found.
        self.letter_coord = {
            'image_dimensions': None,
            'letters': {}
        }

        # Loading the OCR model.
        self.device = device
        self.model = self.load_model(len(class_mapping))

        # Loading the dictionary.
        self.greek_dictionary = self.load_greek_dictionary(
            '/home/ml3/Desktop/Thesis/.venv/Data/filtered_dictionary.dic'
        ) 

        if SEARCH_TEST:
            start = time.time()
            print(self.word_exists("αποτελεσματική"))
            end = time.time()
            legth = end - start
            print(f"Simple search took {legth:.6f} seconds.")
            exit(0)

        # Inverse mapping from index to class name.
        self.classes = sorted(class_mapping.values())
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        # Loading the original image.
        self.original_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

        # Loading edited versions of the input image. 
        self.img_lines = self.input_with_lines(IMG_PATH)

        # Calculating the average space between the
        # notebook lines through a Hough transform.
        self.line_height, self.line_coords = self.hough_distance(self.img_lines)

        # Calculating the coordinates of blocks of text recognized.
        self.block_coords = self.find_blocks(self.original_img)
        
        # Transform applied to each generated cropped image.
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.dysgraphia_features = {
            'baseline_deviation': [],
            'word_size_variation': [],
            'intra_word_spacing': [],
            'slant_consistency': [],
            'random_capitals': []  # Placeholder.
        }

    def load_model(self, num_classes):
        model = OCR(num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
   
        print("Loaded the model!")
        return model
    
    def load_greek_dictionary(self, dict_path):
        """
        Loading the Greek dictionary 
        into a set for faster lookup. 
        Dictionary format: word frequency
        """
        dictionary = set()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                word = parts[0]
                if word:
                    dictionary.add(word)

        print('Loaded the dictionary!')
        return dictionary

    def mark_unknown(self, word, confidence=0):
        if (word.lower() not in self.greek_dictionary) or (confidence < 0.70):
            return f"*{word}*"
        
        return word

    def word_exists(self, word):
        """
        Checks if a word exists in the Greek dictionary.
        Returns True if it exists, False otherwise.
        """
        if (word in self.greek_dictionary):
            return True
        else:
            return False
        
    def input_with_lines(self, IMG_PATH):
        """
        Processes the input image through grayscale conversion,
        adaptive thresholding, and line/character segmentation.

        Returns the original image in a format that is good
        enough for the model to take as an input after each
        character has been isolated.
        """
        # Image loading and grayscale conversion.
        image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

        # Adaptice histogram equalization for contrast.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2 # 11, 2 keeps the notebook lines.

        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        image = 255 - image # Inveting black and white for Hough.

        # Saving the image.
        plt.imshow(image, cmap='gray')
        plt.title("Thresholded Binary Image")
        plt.axis('off')
        plt.savefig("INPUT_WITH_LINES.png", bbox_inches='tight', pad_inches=0)

        return image

# ----------------------------------------------------------------------------#

    '''MIGHT GET REMOVED LATER'''
    def input_without_lines(self, IMG_PATH):
        """
        Processes the input image through grayscale conversion,
        thresholding, and line/character segmentation.
        """
        # Image loading and grayscale conversion.
        img = Image.open(IMG_PATH)
        img_gray = ImageOps.grayscale(img)
        img_array = np.array(img_gray)
        
        # Thresholding.
        threshold = 100
        binary = np.where(img_array <= threshold, 0, 255).astype(np.uint8)

        # Show the image
        plt.imshow(binary, cmap='gray')
        plt.title("Thresholded Binary Image")
        plt.axis('off')
        #plt.show()
        plt.savefig("INPUT_WITHOUT_LINES.png", bbox_inches='tight', pad_inches=0)

        return binary

# ----------------------------------------------------------------------------#

    def black_and_white(self, image_path):
        """
        Converts grayscale image to strict black 
        and white using a threshold of 50.
        """
        image = Image.open(image_path)
        image_gray = ImageOps.grayscale(image)
        image_array = np.array(image_gray)
        threshold = 50
        binary = np.where(image_array <= threshold, 0, 255).astype(np.uint8)
        plt.title("")
        plt.axis('off')
        plt.imshow(binary, cmap='gray')
        plt.savefig("blacknwhite.png", bbox_inches='tight', pad_inches=0)
        return binary
    
    def hough_distance(self, image):
        '''
        Dynamically detecting the lines of the 
        input image using a Hough transform that 
        is robust against duplicate lines.
        '''
        # Parameters for Hough line detection.
        tested_angles = np.linspace(-np.pi/2, np.pi/2, 180, endpoint=False)
        
        # Performing the Hough transform.
        inverted = ~image
        hspace, theta, dist_array = hough_line(inverted, theta=tested_angles) 
        
        # Getting peaks.
        _, angles, distances = hough_line_peaks(
            hspace, theta, dist_array,
            min_distance=20,
            min_angle=5,
            threshold=0.4 * np.max(hspace)
        )
        
        # Convert to numpy arrays.
        if np.isscalar(distances):
            distances = np.array([distances])
        if np.isscalar(angles):
            angles = np.array([angles])
        
        # Sorting lines by distance (y-coordinate).
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_angles = angles[sorted_indices]
        
        # Filtering out nearly identical lines.
        unique_lines = []
        prev_dist = -np.inf
        min_line_separation = max(2, 0.05 * image.shape[0])
        
        for curr_dist, angle in zip(sorted_distances, sorted_angles):
            if abs(curr_dist - prev_dist) >= min_line_separation:
                unique_lines.append((curr_dist, angle))
                prev_dist = curr_dist
        
        # Calculating the average distance between lines.
        line_distances = []
        line_coords = []
        if len(unique_lines) > 1:
            line_distances = [unique_lines[i][0] - unique_lines[i-1][0] 
                            for i in range(1, len(unique_lines))]
        
        avg_line_distance = np.mean(line_distances) if line_distances else 0
        
        # Visualization.
        if True:
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            ax = axes.ravel()
            
            # Input image
            ax[0].imshow(inverted, cmap='gray')
            ax[0].set_title('Input image')
            ax[0].set_axis_off()
            
            hspace_display = np.log(1 + hspace.T) 
            ax[1].imshow(hspace_display, 
                        extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]),
                                dist_array[0], dist_array[-1]],
                        aspect='auto', cmap='gray')
            ax[1].set_title('Hough Transform Space')
            ax[1].set_xlabel('Angle (degrees)')
            ax[1].set_ylabel('Distance (pixels)')
            
            # Detected lines on original image.
            ax[2].imshow(inverted, cmap='gray')
            origin = np.array((0, inverted.shape[1]))
            
            for curr_dist, angle in unique_lines:
                y0, y1 = (curr_dist - origin * np.cos(angle)) / np.sin(angle)
                ax[2].plot(origin, (y0, y1), '-r')
                line_coords.append((str(round(y0, 2)), str(round(y1, 2))))
            
            ax[2].set_xlim(origin)
            ax[2].set_ylim((inverted.shape[0], 0))
            ax[2].set_axis_off()
            ax[2].set_title(f'Detected {len(unique_lines)} lines')
            
            plt.tight_layout()
            plt.show()
            plt.savefig("HOUGH.png", bbox_inches='tight', pad_inches=0)

        # Storing the detected lines.
        self.detected_lines = [dist for dist, _ in unique_lines]
        
        print(f"Average distance between lines: {avg_line_distance:.2f} pixels")
        print(f"Lines detected: {len(unique_lines)}")
        print("LINE COORDS")
        print(line_coords)
        return avg_line_distance, line_coords
    
    def find_blocks(self, logo):
        """
        Creates one bounding box per 1.5 notebook lines.
        For example, for an image with 5 lines (2 of which
        include text), it will create 2 boxes from 0 to 1.5
        and 2 to 3.5.
        """
        VERTICAL_PADDING_RATIO = 0.07  # 7% of line height as padding.
        LINE_EXTENSION_RATIO = 1.5     # Extend the block height by 1.5x of line height.
        MIN_LINE_HEIGHT = 20           # Minimum height to consider a valid line.

        line_ys = sorted([(float(y0),float(y1)) for (y0,y1) in self.line_coords])
        
        # If no lines detected we use the whole image as one block.
        if not line_ys:
            return [[0, logo.shape[0], 0, logo.shape[1]]]
        
        interest = []
        padding = int(self.line_height * VERTICAL_PADDING_RATIO)
        extension = int(self.line_height * LINE_EXTENSION_RATIO)

        # Creating one box per two lines.
        for i in range(0, len(line_ys), 2):
            # Top of area of interest.
            top = int(max(0, max(line_ys[i][0], line_ys[i][1]) + padding))
            
            # Bottom extends to 1.5 lines below.
            if i + 1 < len(line_ys):
                # If there's a next line, extend to midway between current and next.
                bottom = int(min(logo.shape[0], 
                            (max(line_ys[i][0], line_ys[i][1]) + extension)))
            else:
                # For last line, just extend by standard amount
                bottom = int(min(logo.shape[0], 
                            max(line_ys[i][0], line_ys[i][1]) + extension))
            
            # Skipping if the line height is too small.
            if (bottom - top) < MIN_LINE_HEIGHT:
                continue

            interest.append([
                0,             # x1 (start at left edge)
                logo.shape[1], # x2 (end at right edge)
                top,           # y1
                bottom         # y2
            ]) 
        
            # Saving each block as an image.
            block_img= logo[top:bottom, 0:logo.shape[1]]
            block_path = os.path.join(BLOCKS_FOLDER, f"block_{i//2 + 1}.png")
            cv2.imwrite(block_path, block_img)

        # Visualization.
        color_logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR)
        
        # Drawing text blocks over areas of interest.
        for i, (x1, x2, y1, y2) in enumerate(interest): #enumerate line boundaries 
            cv2.rectangle(color_logo, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(color_logo, f"Block {i+1}", (x1 + 10, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(color_logo, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        plt.imsave("BLOCKS.png", color_logo)

        return interest
    
# ----------------------------------------------------------------------------#

    def _get_expected_baseline(self, y_position):
        """
        Finds the closest baseline for a given y-coordinate.
        """
        if not hasattr(self, 'detected_lines') or not self.detected_lines:
            return None
            
        # Finds the nearest baseline.
        closest_line = min(self.detected_lines, key=lambda line: abs(line - y_position))
        return closest_line
    
    def analyze_handwriting_consistency(self):
        """
        Calculate consistency metrics for dysgraphia detection
        """
        metrics = {}
        
        # 1. Baseline Stability Analysis.
        if self.dysgraphia_features['baseline_deviation']:
            avg_deviation = np.mean(self.dysgraphia_features['baseline_deviation'])
            dev_std = np.std(self.dysgraphia_features['baseline_deviation'])
            metrics['baseline_stability'] = {
                'mean_deviation': avg_deviation,
                'deviation_std': dev_std,
                'consistency_score': max(0, 1 - (avg_deviation / (self.line_height or 1)))
            }
        
        # 2. Word Size Consistency.
        if self.dysgraphia_features['word_size_variation']:
            size_std = np.std(self.dysgraphia_features['word_size_variation'])
            metrics['size_consistency'] = {
                'size_std': size_std,
                'consistency_score': max(0, 1 - (size_std / (self.line_height or 1)))
            }
        
        # 3. Intra-word Spacing Analysis (Requires character segmentation)
        # [To be implemented in character segmentation phase]
        
        # 4. Slant Analysis (Requires character-level segmentation)
        # [To be implemented later]
        
        return metrics
    
# ----------------------------------------------------------------------------#

    def recognize_character(self, char_image):
        """
        Recognizing a single character.
        Returns the character itself and the 
        respective confidence of the prediction.
        """
        # Original image.
        plt.figure(figsize=(10, 4))
        plt.style.use('dark_background')
        plt.subplot(1, 2, 1)
        plt.imshow(char_image, cmap='gray')
        plt.title("Original Character")
        plt.axis('off')

        # Transforming and preparing for the model.
        char_pil = Image.fromarray(char_image).convert('L')
        transformed_img = self.transform(char_pil)
        
        # Preparing transformed image for display.
        img_to_show = transformed_img.squeeze().cpu().numpy()
        img_to_show = (img_to_show * 0.5) + 0.5  # Undoing normalization.
        img_to_show = np.clip(img_to_show, 0, 1)
        
        # Transformed image
        plt.subplot(1, 2, 2)
        plt.imshow(img_to_show, cmap='gray')
        plt.title("Transformed (Model Input)")
        plt.axis('off')
        plt.style.use('dark_background')

        plt.tight_layout()
        plt.show()
        
        # Continuing with recognition.
        char_tensor = transformed_img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(char_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            char = self.idx_to_class[pred.item()]
            
            return char.lower(), conf.item()
    
    def reconstruct_text(self, charPred, spaces):
        """
        Reconstructs the recognized characters into words and 
        lines, using the space information and dictionary lookup.
        """
        full_text = []
        current_word = []
        current_confidences = []
        confidence_scores = []
        
        co = 0
        for char in charPred:
            # Space between words.
            if spaces[co] == 1:  
                if current_word:
                    word = ''.join(current_word)
                    avg_conf = sum(current_confidences)/len(current_confidences) if current_confidences else 0
                    marked_word = self.mark_unknown(word, avg_conf)
                    full_text.append(marked_word)
                    current_word = []
                    current_confidences = []
                full_text.append(' ')
            # New line.
            elif spaces[co] == 2:  
                if current_word:
                    word = ''.join(current_word)
                    avg_conf = sum(current_confidences)/len(current_confidences) if current_confidences else 0
                    marked_word = self.mark_unknown(word, avg_conf)
                    full_text.append(marked_word)
                    current_word = []
                    current_confidences = []
                full_text.append('\n')
            else:
                current_word.append(char)
                current_confidences.append(1.0)  # Placeholder confidence.
            
            co += 1
        
        # Adds the last word if it exists.
        if current_word:
            word = ''.join(current_word)
            avg_conf = sum(current_confidences)/len(current_confidences) if current_confidences else 0
            marked_word = self.mark_unknown(word, avg_conf)
            full_text.append(marked_word)
        
        # Calculating statistics!
        text = ''.join(full_text).strip()
        words = [w for w in text.split() if not w.startswith('\n')]
        known_words = [w for w in words if not w.startswith('*')]
        accuracy = len(known_words)/len(words) if words else 0
        
        return {
            'text': text,
            'accuracy': accuracy,
            'confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'char_count': len(text.replace('\n', '')),
            'word_count': len(words),
            'unknown_words': [w.strip('*') for w in words if w.startswith('*')]
        }
    
    def recognize_text(self):
        """
        Main OCR pipeline that processes an image file and
        returns the recognized text with statistics. 
        """
        # Segments words and saves them as PNGs in LetterDump.
        # TODO: pipeline with segment.py
        print("Character segmentation completed.")
        
        # ???

        # Printing the coordinates of each word found.
        for filename, coords in recognizer.letter_coord['letters'].items():
            print(f"{filename}: x={coords[0]}, y={coords[1]}")
        
        # Recognizing each character added to the image folder.
        charPred = []
        char_data = []
        for i in sorted(os.listdir(DUMP+"/")):
            img_path = os.path.join(DUMP, i)
            with Image.open(img_path) as image:
                char, conf = self.recognize_character(np.asarray(image))
                char_data.append((char, conf))
            #os.remove(img_path)
        
        # Dysgraphia analysis.
        dysgraphia_metrics = self.analyze_handwriting_consistency()
        result = self.reconstruct_text(charPred, self.spaces)
        result['dysgraphia_metrics'] = dysgraphia_metrics
    
        # Reconstructing the text with dictionary checking.
        return self.reconstruct_text(charPred, self.spaces)

# ----------------------------------------------------------------------------#

if __name__ == '__main__':
    recognizer = GreekTextRecognizer("cuda")
    result = recognizer.recognize_text()
    
    print("\nFinal Results:")
    print(f"Recognized Text: {result['text']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Character Count: {result['char_count']}")
    print(f"Word Count: {result['word_count']}")
    print(f"Unknown Words: {result['unknown_words']}")

    print("\nDysgraphia Analysis:")
    for metric, values in result['dysgraphia_metrics'].items():
        print(f"{metric.upper()}:")
        for k, v in values.items():
            print(f"  {k}: {v:.4f}")

# ----------------------------------------------------------------------------#
