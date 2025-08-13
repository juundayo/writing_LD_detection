# ----------------------------------------------------------------------------#

import os
import numpy as np
import torch
import cv2
import time
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from torchvision import transforms
from skimage.transform import (hough_line, hough_line_peaks)
from PIL import Image, ImageOps
from skimage import io
from collections import defaultdict

from vt_model import OCR
from class_renaming import class_mapping
from segmentation import process_image_block, get_image_average

# ----------------------------------------------------------------------------#

MODEL_PATH = "/home/ml3/Desktop/Thesis/Models/rs1_tt2_v2.pth"

LETTER_FOLDER = "/home/ml3/Desktop/Thesis/LetterFolder"
BLOCKS_FOLDER = "/home/ml3/Desktop/Thesis/BlockImages"
WRITING_DIS_FOLDER = "/home/ml3/Desktop/Thesis/WritingDisorder"

IMG_PATH = "/home/ml3/Desktop/Thesis/Screenshot_15.png"
#IMG_PATH = "/home/ml3/Desktop/Thesis/two_mimir.jpg"
IMG_HEIGHT = 512
IMG_WIDTH = 78

SEARCH_TEST = False

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

class GreekTextRecognizer:
    def __init__(self, device):        
        # Loading the OCR model.
        self.device = device
        self.classes = class_mapping.values()
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        self.model = self.load_model(len(self.classes))

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

        # Adaptive histogram equalization for contrast.
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
            min_distance=35,
            min_angle=10,
            threshold=0.4 * np.max(hspace)
        )
        
        # Converting to numpy arrays.
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

        line_coords = sorted(line_coords, key=lambda coord: float(coord[0]), reverse=True)
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
        LINE_EXTENSION_RATIO = 1.33    # Extend the block height by 0.5 of line height.
        MIN_LINE_HEIGHT = 20           # Minimum height to consider a valid line.

        line_ys = sorted([(float(y0), float(y1)) for (y0, y1) in self.line_coords])     
           
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

    def visualize_baseline_deviation(self, block_num, word_data):
        """
        Visualizes baseline deviation for words in a block compared to notebook lines.
        Marks potential dysgraphia if deviation exceeds dynamic threshold.
        Saves the plot as 'baseline_deviation_block_{block_num}.png'
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get line positions for this block
        line_ys = sorted([max(float(y0), float(y1)) for (y0, y1) in self.line_coords])
        block_top = (block_num - 1) * 1.5 * self.line_height
        block_bottom = block_num * 1.5 * self.line_height
        
        # Filter lines within this block's vertical range
        block_lines = [y for y in line_ys if block_top <= y <= block_bottom]
        
        if not block_lines:
            print(f"No lines detected in block {block_num}")
            return
        
        # Calculate dynamic threshold (30% of line height)
        threshold = 0.3 * self.line_height
        
        # Plot notebook lines
        for line_y in block_lines:
            ax.axhline(y=line_y, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=line_y + self.line_height, color='gray', linestyle='--', alpha=0.5)
        
        # Track baseline deviations
        deviations = []
        word_positions = []
        word_texts = []
        
        # Process each word
        for word_idx, word in enumerate(word_data):
            if not word['characters']:
                continue
                
            # Calculate word baseline (bottom of first character)
            first_char = word['characters'][0]
            baseline_y = first_char[3]  # cy2 (bottom y-coordinate)
            
            # Find nearest notebook line below this baseline
            line_below = min([y for y in block_lines if y >= baseline_y], default=None)
            if line_below is None:
                line_below = max(block_lines)
            
            # Calculate deviation from line
            deviation = abs(baseline_y - line_below)
            deviations.append(deviation)
            word_positions.append(word_idx)
            word_texts.append(f"Word {word_idx+1}")
            
            # Mark if deviation exceeds threshold
            if deviation > threshold:
                ax.plot(word_idx, baseline_y, 'ro', markersize=8)
                ax.text(word_idx, baseline_y + 5, "!", 
                        color='red', ha='center', fontsize=12, weight='bold')
        
        # Plot deviation curve
        ax.plot(word_positions, [w['characters'][0][3] if w['characters'] else 0 for w in word_data], 
                'b-', label='Word Baselines')
        
        # Plot threshold zone (between lines)
        for line_y in block_lines:
            ax.add_patch(Rectangle((0, line_y - threshold), 
                        len(word_data), 2*threshold,
                        alpha=0.2, color='green', 
                        label='Normal Zone' if line_y == block_lines[0] else ""))
        
        # Style plot
        ax.set_title(f"Block {block_num} - Baseline Deviation Analysis")
        ax.set_xlabel("Word Position")
        ax.set_ylabel("Vertical Position (pixels)")
        ax.set_xticks(word_positions)
        ax.set_xticklabels(word_texts, rotation=45)
        ax.legend()
        ax.grid(True)
        
        # Checking for potential dysgraphia
        dysgraphia_detected = any(d > threshold for d in deviations)
        if dysgraphia_detected:
            ax.text(0.5, 0.95, "Potential Dysgraphia Detected: Irregular Baselines", 
                    transform=ax.transAxes, color='red', 
                    ha='center', fontsize=12, weight='bold')
            print(f"Block {block_num}: Potential dysgraphia detected - baseline deviations exceed threshold")
        
        plt.tight_layout()
        
        # Saving the figure.
        output_path = os.path.join(WRITING_DIS_FOLDER, f"baseline_deviation_block_{block_num}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved baseline deviation plot to {output_path}")
        
        plt.close()
            
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
        os.makedirs(WRITING_DIS_FOLDER, exist_ok=True)

        # Clearning previous blocks and letter crops.
        for folder in [BLOCKS_FOLDER, LETTER_FOLDER, WRITING_DIS_FOLDER]:
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
            self.visualize_baseline_deviation(block_idx + 1, word_data)

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

        # Dictionary to store recognized text by block and word.
        recognized_text = defaultdict(lambda: defaultdict(list))
        char_predictions = {}

        for char_file in os.listdir(LETTER_FOLDER):
            parts = char_file.split('_')
            block_num = int(parts[1])    # block_1_word_1_char_2.png -> block_1
            word_num = int(parts[3])     # block_1_word_1_char_2.png -> word_1
            char_num = int(parts[5].split('.')[0]) # [..]_char_2.png -> char_2

            char_path = os.path.join(LETTER_FOLDER, char_file)
            char_img = Image.open(char_path).convert('L')
            char_tensor = self.transform(char_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(char_tensor)
                _, predicted = torch.max(output.data, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()
                predicted_char = self.idx_to_class[predicted.item()]
                
                if predicted_char in SIMILAR_PAIRS and predicted_char.isupper():
                    predicted_char = SIMILAR_PAIRS[predicted_char]
                
                char_predictions[(block_num, word_num, char_num)] = (predicted_char, confidence)
            
            max_block = max([k[0] for k in char_predictions.keys()]) if char_predictions else 0
            
            # Getting all words in this block.
            for block_num in range(1, max_block + 1):
                block_words = [k[1] for k in char_predictions.keys() if k[0] == block_num]
                max_word = max(block_words) if block_words else 0

                # Getting all characters for this word, sorted by position.
                for word_num in range(1, max_word + 1):
                    word_chars = []
                    word_keys = [k for k in char_predictions.keys() 
                                 if k[0] == block_num and k[1] == word_num]
                    word_keys.sort(key=lambda x: x[2]) # Sorting by character number.

                    for key in word_keys:
                        char, confidence = char_predictions[key]
                        word_chars.append(char)
                    
                    if word_chars:
                        recognized_text[block_num][word_num] = ''.join(word_chars)
            
            final_output = []
            for block_num in sorted(recognized_text.keys()):
                block_words = []
                prev_word_num = 0

                for word_num in sorted(recognized_text[block_num].keys()):
                    # Adding space if there are missing word numbers.
                    if word_num > prev_word_num + 1:
                        block_words.append('')
                    
                    word_text = recognized_text[block_num][word_num]
                    block_words.append(word_text)
                    prev_word_num = word_num
                
                block_sentence = ' '.join(block_words)
                final_output.append(block_sentence)
            
        print("\nRecognized Text:")
        for i, sentence in enumerate(final_output, 1):
            print(f"Block {i}: {sentence}")
            
# ----------------------------------------------------------------------------#

if __name__ == '__main__':
    recognizer = GreekTextRecognizer("cuda" if torch.cuda.is_available() else "cpu")
    result = recognizer.recognize_text()
    
# ----------------------------------------------------------------------------#
