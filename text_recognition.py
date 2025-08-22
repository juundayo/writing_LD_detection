# ----------------------------------------------------------------------------#

import os
import numpy as np
import torch
import cv2
import time
import math
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
from segmentation import process_image_block

# ----------------------------------------------------------------------------#

#MODEL_PATH = "/home/ml3/Desktop/Thesis/Models/rs1_tt2_v2.pth"
MODEL_PATH = "/home/ml3/Desktop/Thesis/Models/822_newdata.pth"

LETTER_FOLDER = "/home/ml3/Desktop/Thesis/LetterFolder"
BLOCKS_FOLDER = "/home/ml3/Desktop/Thesis/BlockImages"
WRITING_DIS_FOLDER = "/home/ml3/Desktop/Thesis/WritingDisorder"

IMG_PATH = "/home/ml3/Desktop/Thesis/two_mimir.jpg"
#IMG_PATH = "/home/ml3/Desktop/Thesis/IMG_20250809_113620.jpg"
#IMG_PATH = "/home/ml3/Desktop/Thesis/0_Test2.jpg"
#IMG_PATH = "/home/ml3/Desktop/Thesis/.venv/Screenshot_17.png"
#IMG_PATH = "/home/ml3/Desktop/Thesis/.venv/caps.jpg"
#IMG_PATH = "/home/ml3/Desktop/Thesis/0_Test1.jpg"

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

'''
'Μ': 'μ', 'μ': 'Μ',
'''

# ----------------------------------------------------------------------------#

class GreekTextRecognizer:
    def __init__(self, device):        
        # Loading the OCR model.
        self.device = device
        self.classes = class_mapping.values()
        print(f"Number of classes: {len(self.classes)}")
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
    
    def slant_correct_character(self, char_img):
        """
        Corrects slant in an individual character image by detecting vertical strokes
        and rotating to make them upright.
        """
        gray = char_img.copy()

        # Binarize with Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Edge detection for vertical stroke detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # Hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=10)
        if lines is None:
            return char_img  # No correction if no lines found

        # Select near-vertical lines (80°–100°)
        vertical_angles = []
        for rho, theta in lines[:, 0]:
            angle_deg = np.degrees(theta)
            if 80 < angle_deg < 100:
                vertical_angles.append(theta)

        if not vertical_angles:
            return char_img

        # Compute rotation needed
        avg_theta = np.mean(vertical_angles)
        angle_correction = (np.pi / 2) - avg_theta
        angle_correction_deg = np.degrees(angle_correction)

        # Rotate around image center
        h, w = char_img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle_correction_deg, 1.0)
        corrected = cv2.warpAffine(char_img, rot_mat, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        return corrected

    def hough_distance(self, image, SHOW_PLOTS=True):
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
        
        # Normalize angles to consistent range (-90° to 90°)
        normalized_angles = []
        normalized_distances = []
        
        for dist, angle in zip(distances, angles):
            angle_deg = np.rad2deg(angle)
            
            # Convert near-vertical angles to negative range
            if angle_deg > 45:  # If angle is between 45° and 90°
                angle_deg -= 180  # Convert to negative equivalent
                dist = -dist     # Flip distance sign to maintain line position
            
            normalized_angles.append(np.deg2rad(angle_deg))
            normalized_distances.append(dist)
        
        # Now use normalized_angles and normalized_distances for further processing
        angles = np.array(normalized_angles)
        distances = np.array(normalized_distances)
        
        # Sorting lines by distance (y-coordinate).
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_angles = angles[sorted_indices]

        # Estimating the average line spacing.
        if len(sorted_distances) > 1:
            rough_diffs = np.diff(sorted_distances)
            avg_line_distance = np.median(rough_diffs)
        else:
            avg_line_distance = 20  # Fallback for single-line cases.
        
        # Adaptive minimum line separation.
        min_line_separation = max(2, 0.6 * avg_line_distance)

        # Filtering out nearly identical lines.
        unique_lines = []
        prev_dist = -np.inf
        
        for curr_dist, angle in zip(sorted_distances, sorted_angles):
            if abs(curr_dist - prev_dist) >= min_line_separation:
                unique_lines.append((curr_dist, angle))
                prev_dist = curr_dist

        line_coords = []
        line_distances = []
        if len(unique_lines) > 1:
            line_distances = [unique_lines[i][0] - unique_lines[i-1][0] 
                            for i in range(1, len(unique_lines))]
                    
        avg_line_distance = np.mean(line_distances) if line_distances else 0
        
        # Visualization.
        if SHOW_PLOTS == True:
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
            plt.close()
        
        line_coords = sorted(line_coords, key=lambda coord: float(coord[0]), reverse=True)
        return avg_line_distance, line_coords
    
    def find_blocks(self, logo):
        """
        Creates one bounding box per 1.5 notebook lines.
        For example, for an image with 5 lines (2 of which
        include text), it will create 2 boxes from 0 to 1.5
        and 2 to 3.5.
        """
        VERTICAL_PADDING_RATIO = 0.2   # 30% of line height as padding.
        LINE_EXTENSION_RATIO = 1.33    # Extend the block height by 1.33 of line height.
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

    def wd1_baseline(self, block_num, word_data):
        """
        Analyzes baseline consistency using the existing hough_distance() function
        but applied to individual blocks for more accurate local baselines.
        """
        # Load the block image
        block_path = os.path.join(BLOCKS_FOLDER, f"block_{block_num}.png")
        block_img = cv2.imread(block_path, cv2.IMREAD_GRAYSCALE)
        if block_img is None:
            return {'is_problematic': False}
        
        # Create visualization figure
        _, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(block_img, cmap='gray')
        
        # 1.  HOUGH LINES FOR THIS BLOCK -------------------------------------
        # We need to temporarily modify the image for Hough detection
        processed_block = self.input_with_lines(block_path)  # Reuse your preprocessing
        _, block_line_coords = self.hough_distance(processed_block)  # Get lines for just this block
        
        # 2. DETERMINING REFERENCE BASELINE ----------------------------------
        if block_line_coords:
            # Get the bottom-most line (maximum y-coordinate)
            bottom_line = max(block_line_coords, key=lambda coord: max(float(coord[0]), float(coord[1])))
            y0, y1 = float(bottom_line[0]), float(bottom_line[1])
            baseline_y = max(y0, y1)  # Use the lower of the two y-coordinates
            
            # Draw the reference line across the full block width
            ax.axhline(y=baseline_y, color='blue', linestyle='--', 
                    linewidth=2, alpha=0.7, label='Notebook Baseline')
        else:
            # Fallback: use bottom 10% of block as pseudo-baseline
            baseline_y = 0.9 * block_img.shape[0]
            ax.axhline(y=baseline_y, color='blue', linestyle='--', 
                    linewidth=2, alpha=0.7, label='Estimated Baseline')
        
        # 3. ANALYZING WORD BASELINE DEVIATIONS ----------------------------
        DEVIATION_THRESHOLD = 0.2 * self.line_height  # 20% of standard line height
        problematic_words = []
        
        for word_idx, word in enumerate(word_data):
            if not word['characters']:
                continue
                
            char_tops = [char[1] for char in word['characters']]     # cy1 values (top)
            char_bottoms = [char[3] for char in word['characters']]  # cy2 values (bottom)

            # Estimate median baseline ignoring deep descenders
            median_height = np.median([b - t for t, b in zip(char_tops, char_bottoms)])
            descender_limit = np.median(char_tops) + median_height * 1.1  # 11% tolerance.

            filtered_bottoms = [b for b in char_bottoms if b <= descender_limit]
            if not filtered_bottoms:
                filtered_bottoms = char_bottoms  # Fallback if all were descenders.

            word_baseline = np.median(filtered_bottoms)

            deviation = abs(word_baseline - baseline_y)
            
            # Word bounding box coordinates.
            x_start = min(char[0] for char in word['characters'])
            x_end = max(char[2] for char in word['characters'])
            
            if deviation > DEVIATION_THRESHOLD:
                problematic_words.append((word_idx, deviation))
                
                # Drawing problematic word baseline in red.
                ax.plot([x_start, x_end], [word_baseline, word_baseline],
                    'r-', linewidth=2, alpha=0.6, label='Deviated Baseline' if word_idx == 0 else "")
                
                # Drawing deviation measurement.
                mid_x = (x_start + x_end) / 2
                ax.plot([mid_x, mid_x], [baseline_y, word_baseline],
                    'r:', linewidth=1, alpha=0.4)
                ax.text(mid_x, (baseline_y + word_baseline)/2,
                    f"{deviation:.1f}px", color='red', 
                    ha='center', va='center')
            else:
                # Drawing normal word baseline in green.
                ax.plot([x_start, x_end], [word_baseline, word_baseline],
                    'g-', linewidth=1, alpha=0.4, label='Normal Baseline' if word_idx == 0 else "")
        
        # 4. VISUALIZATION AND OUTPUT --------------------------------------
        is_problematic = len(problematic_words) > 0
        
        # Only show legend if we have both types of baselines.
        if problematic_words and len(problematic_words) < len(word_data):
            ax.legend()
        
        ax.set_title(f"Block {block_num} - Baseline Analysis\n"
                    f"Reference: y={baseline_y:.1f}px | "
                    f"Threshold: ±{DEVIATION_THRESHOLD:.1f}px | "
                    f"{'⚠️ Potential Writing Disorder' if is_problematic else '✓ Normal'}")
        ax.axis('off')
        
        # Saving the visualization.
        output_path = os.path.join(WRITING_DIS_FOLDER, 
                                f"baseline_block_{block_num}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Debugging analysis results.
        return {
            'block_num': block_num,
            'is_problematic': is_problematic,
            'problematic_words': len(problematic_words),
            'max_deviation': max(d[1] for d in problematic_words) if problematic_words else 0,
            'reference_baseline': baseline_y,
            'deviation_threshold': DEVIATION_THRESHOLD
        }

    def wd2_uppercase_lowercase(self, block_num, word_data):
        """
        Flags words with more than one capital letter (excluding the first letter).
        Parameters:
            block_num (int): Current block number
            word_data (list): List of word dicts with 'characters' and 'predictions'
        Returns:
            dict: Results with flagged words
        """
        flagged_words = []

        for w_idx, word in enumerate(word_data):
            if not word.get('predictions'):
                continue

            predicted_chars = [p[0] for p in word['predictions']]  # (char, conf)
            capitals = [c for i, c in enumerate(predicted_chars) if c.isupper() and i != 0]

            if len(capitals) > 0:
                flagged_words.append({
                    'word_index': w_idx,
                    'word_text': ''.join(predicted_chars),
                    'extra_capitals': capitals
                })

        return {
            'block_num': block_num,
            'wd2_flagged': flagged_words,
            'wd2_issue_count': len(flagged_words)
        }
    
    def wd3_spaces(self, block_num, word_data):
        """
        Flags potentially problematic spacing between words.
        Uses bounding boxes to measure horizontal gaps.
        Parameters:
            block_num (int)
            word_data (list): Each word has 'characters' with bounding boxes
        Returns:
            dict: Results with flagged gaps
        """
        word_positions = []
        for w_idx, word in enumerate(word_data):
            if not word.get('characters'):
                continue
            x_start = min(c[0] for c in word['characters'])  # Left.
            x_end = max(c[2] for c in word['characters'])    # Right.
            word_positions.append((w_idx, x_start, x_end))

        if len(word_positions) < 2:
            return {'block_num': block_num, 'wd3_flagged': [], 'wd3_issue_count': 0}

        # Sorting by position.
        word_positions.sort(key=lambda x: x[1])

        # Computing gaps.
        gaps = []
        for i in range(len(word_positions) - 1):
            gap = word_positions[i + 1][1] - word_positions[i][2]
            gaps.append(gap)

        if not gaps:
            return {'block_num': block_num, 'wd3_flagged': [], 'wd3_issue_count': 0}

        median_gap = np.median(gaps)
        min_gap_thresh = 0.75 * median_gap
        max_gap_thresh = 2.0 * median_gap

        flagged_gaps = []
        for i, gap in enumerate(gaps):
            if gap < min_gap_thresh or gap > max_gap_thresh:
                flagged_gaps.append({
                    'between_words': (word_positions[i][0], word_positions[i + 1][0]),
                    'gap': gap,
                    'median_gap': median_gap
                })

        return {
            'block_num': block_num,
            'wd3_flagged': flagged_gaps,
            'wd3_issue_count': len(flagged_gaps)
        }

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
        block_word_data = {}
        for block_idx, block_file in enumerate(block_files):
            block_path = os.path.join(BLOCKS_FOLDER, block_file)
            block_img = cv2.imread(block_path)

            # Getting the Hough line above this block (to extend characters upwards).
            current_block_top = int(block_file.split('_')[1].split('.')[0]) * self.line_height
            upper_line_y = self.find_nearest_above_line(current_block_top)

            # Segmenting the block into words and characters.
            _, word_data = process_image_block(block_img)
            block_word_data[block_idx + 1] = word_data

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

            char_img = cv2.imread(char_path, cv2.IMREAD_GRAYSCALE)
            #char_img = self.slant_correct_character(char_img) # Character slant correction.
            char_pil = Image.fromarray(char_img)
            char_tensor = self.transform(char_pil).unsqueeze(0).to(self.device)

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

        # Writing disorder analysis.
        for block_num in block_word_data.keys():
            if block_num in recognized_text:
                # Preparing word data with predictions for writing disorder analysis.
                word_data_with_predictions = []
                for word_idx, word in enumerate(block_word_data[block_num]):
                    if (word_idx + 1) in recognized_text[block_num]:
                        word_with_pred = word.copy()
                        # Getting predictions for this word.
                        word_keys = [k for k in char_predictions.keys() 
                                    if k[0] == block_num and k[1] == word_idx + 1]
                        word_keys.sort(key=lambda x: x[2])
                        predictions = []
                        for key in word_keys:
                            char, confidence = char_predictions[key]
                            predictions.append((char, confidence))
                        word_with_pred['predictions'] = predictions
                        word_data_with_predictions.append(word_with_pred)
                    else:
                        word_data_with_predictions.append(word)
                
                # Calling all writing disorder functions.
                print(f"\n=== Writing Disorder Analysis for Block {block_num} ===")
                
                # WD1: Baseline analysis.
                wd1_result = self.wd1_baseline(block_num, word_data_with_predictions)
                print(f"WD1 - Baseline Issues: {wd1_result['is_problematic']}")
                if wd1_result['is_problematic']:
                    print(f"   Problematic words: {wd1_result.get('problematic_words', 'N/A')}")
                    print(f"   Max deviation: {wd1_result.get('max_deviation', 'N/A'):.1f}px")
                print()

                # WD2: Uppercase/Lowercase analysis.
                wd2_result = self.wd2_uppercase_lowercase(block_num, word_data_with_predictions)
                print(f"WD2 - Uppercase Issues: {wd2_result['wd2_issue_count']}")
                if wd2_result['wd2_flagged']:
                    for flagged_word in wd2_result['wd2_flagged']:
                        print(f"   Word {flagged_word['word_index']}: '{flagged_word['word_text']}' "
                            f"with extra capitals: {flagged_word['extra_capitals']}")
                print()

                # WD3: Spacing analysis.
                wd3_result = self.wd3_spaces(block_num, word_data_with_predictions)
                print(f"WD3 - Spacing Issues: {wd3_result['wd3_issue_count']}")
                if wd3_result['wd3_flagged']:
                    for gap in wd3_result['wd3_flagged']:
                        print(f"   Gap between words {gap['between_words']}: {gap['gap']:.1f}px "
                            f"(median: {gap['median_gap']:.1f}px)")
                print()

                # Summary
                total_issues = (1 if wd1_result['is_problematic'] else 0) + \
                            wd2_result['wd2_issue_count'] + \
                            wd3_result['wd3_issue_count']
                print(f"Total Writing Disorder Indicators: {total_issues}")

            # Final text assembly.
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
