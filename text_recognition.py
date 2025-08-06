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
from segmentation import process_image_block, get_image_average, Rectangle, Word

# ----------------------------------------------------------------------------#

MODEL_PATH = "/home/ml3/Desktop/Thesis/Models/rs1_tt2_v2.pth"

LETTER_FOLDER = "/home/ml3/Desktop/Thesis/LetterFolder"
BLOCKS_FOLDER = "/home/ml3/Desktop/Thesis/BlockImages"

IMG_PATH = "/home/ml3/Desktop/Thesis/two_mimir.jpg"
IMG_HEIGHT = 512
IMG_WIDTH = 78

SEARCH_TEST = False

# ----------------------------------------------------------------------------#

class GreekTextRecognizer:
    def __init__(self, device):        
        # Loading the OCR model.
        self.device = device
        #self.model = self.load_model(len(class_mapping))

        self.letter_coord = {
            'image_dimensions': None,
            'letters': {}
        }

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
        #self.classes = sorted(class_mapping.values())
        #self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        # Loading the original image.
        self.coloured_original = cv2.imread(IMG_PATH)
        self.original_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

        # Loading edited versions of the input image. 
        self.img_lines = self.input_with_lines(IMG_PATH)

        # Calculating the average space between the
        # notebook lines through a Hough transform.
        self.line_height, self.line_coords = self.hough_distance(self.img_lines)
        
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
        VERTICAL_PADDING_RATIO = 0.1  # 7% of line height as padding.
        LINE_EXTENSION_RATIO = 1.33    # Extend the block height by 1.25x of line height.
        MIN_LINE_HEIGHT = 20           # Minimum height to consider a valid line.

        line_ys = sorted([(float(y0),float(y1)) for (y0,y1) in self.line_coords])
        
        # If no lines detected we use the whole image as one block.
        if not line_ys:
            return [[0, logo.shape[0], 0, logo.shape[1]]]
        
        interest = []
        padding_above = int(self.line_height * VERTICAL_PADDING_RATIO)
        padding_below = int(self.line_height * LINE_EXTENSION_RATIO)

        # Creating one box per two lines.
        for i in range(0, len(line_ys), 2):
            # Top of area of interest.
            top = int(max(0, max(line_ys[i][0], line_ys[i][1]) - padding_above))
            
            # Bottom extends to 1.5 lines below.
            if i + 1 < len(line_ys):
                # If there's a next line, extend to midway between current and next.
                bottom = int(min(logo.shape[0], 
                            (max(line_ys[i][0], line_ys[i][1]) + padding_below)))
            else:
                # For last line, just extend by standard amount
                bottom = int(min(logo.shape[0], 
                            max(line_ys[i][0], line_ys[i][1]) + padding_below))
            
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
        
    def find_nearest_above_line(self, y_position):
        """Finds the nearest Hough line above the given y-position."""
        # Converting line_coords to y positions (taking the lower of the two y values).
        line_ys = []
        for y0, y1 in self.line_coords:
            line_ys.append(max(float(y0), float(y1)))
        
        # Filtering lines above current position.
        above_lines = [line for line in line_ys if line < y_position]
        
        if not above_lines:
            return None
            
        # Returning the closest line above.
        return max(above_lines)

    def recognize_text(self):
        """
        Main OCR pipeline that processes an image file 
        and returns the recognized text with statistics. 
        """
        os.makedirs(BLOCKS_FOLDER, exist_ok=True)
        os.makedirs(LETTER_FOLDER, exist_ok=True)

        # Clearning previous blocks and letter crops.
        for folder in [BLOCKS_FOLDER, LETTER_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Extracting blocks of text from the image.
        self.find_blocks(self.original_img)
        block_files = sorted([f for f in os.listdir(BLOCKS_FOLDER) if f.startswith('block_')],
                        key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Processing each block of text.
        for block_idx, block_file in enumerate(block_files):
            block_path = os.path.join(BLOCKS_FOLDER, block_file)
            block_img = cv2.imread(block_path)

            # Getting the Hough line above this block (to extend characters upwards).
            current_block_top = int(block_file.split('_')[1].split('.')[0]) * self.line_height
            upper_line_y = self.find_nearest_above_line(current_block_top)

            # Getting the average block size for filtering.
            im_average = get_image_average(self.coloured_original)

            # Segmenting the block into words and characters.
            _, word_data = process_image_block(block_img, im_average)

            # Processing each word and character in the block.
            for word_idx, word in enumerate(word_data):
                for char_idx, char_bbox in enumerate(word['characters']):
                    cx, cy, cx2, cy2 = char_bbox

                    if upper_line_y is not None:
                        adjusted_cy = max(0, upper_line_y - current_block_top)
                    else:
                        adjusted_cy = 0

                    # Extending the character crop upwards.
                    char_crop = block_img[adjusted_cy:cy2, cx:cx2]

                    char_filename = f"block_{block_idx+1}_word_{word_idx+1}_char_{char_idx+1}.png"

                    char_path = os.path.join(LETTER_FOLDER, char_filename)
                    cv2.imwrite(char_path, char_crop)

                    
# ----------------------------------------------------------------------------#

if __name__ == '__main__':
    recognizer = GreekTextRecognizer("cuda" if torch.cuda.is_available() else "cpu")
    result = recognizer.recognize_text()
    
# ----------------------------------------------------------------------------#
