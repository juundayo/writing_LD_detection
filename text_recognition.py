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

VENV_PATH = "/home/ml3/Desktop/CRAFT/.venv"
CRAFT_PATH = "/home/ml3/Desktop/CRAFT/.venv/CRAFT-pytorch-master/craft_test.py"
CRAFT_RESULTS_PATH = "/home/ml3/Desktop/CRAFT/.venv/outputs/image_text_detection.txt"

IMG_PATH = "/home/ml3/Desktop/Thesis/two_mimir.jpg"
#IMG_PATH = "/home/ml3/Downloads/input3.jpg"

DUMP = "/home/ml3/Desktop/Thesis/LetterDump"
OUTPUT_FOLDER = "/home/ml3/Desktop/Thesis/LetterCrops"

IMG_HEIGHT = 512
IMG_WIDTH = 78

SEARCH_TEST = False

# ----------------------------------------------------------------------------#

class GreekTextRecognizer:
    def __init__(self, device):
        # Creating the generated cropped image folder. 
        os.makedirs(DUMP, exist_ok=True)

        # Removing all the previous files from the folder.
        for filename in os.listdir(DUMP):
            file_path = os.path.join(DUMP, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        
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

        # Loading edited versions of the input image. 
        self.img_lines = self.input_with_lines(IMG_PATH)
        self.img_no_lines = self.input_without_lines(IMG_PATH)

        # Calculating the space between the notebook lines.
        self.line_height, self.line_coords = self.line_distance(self.img_lines)

        # Calculating the coordinates of blocks of text recognized.
        #self.block_coords = self.find_blocks(self.img_no_lines)
        self.block_coords = self.find_blocks(self.img_lines)

        # Detecting space boxes for characters.
        self.spaces = self.detect_spaces(self.img_no_lines, self.block_coords)
        
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

        # Sharpening.
        #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #image = cv2.filter2D(image, -1, kernel)
        
        image = 255 - image # Inveting black and white.

        # Show the image
        plt.imshow(image, cmap='gray')
        plt.title("Thresholded Binary Image")
        plt.axis('off')
        plt.savefig("INPUT_WITH_LINES.png", bbox_inches='tight', pad_inches=0)

        return image
    
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

    def black_and_white(self, image):
        """
        Converts grayscale image to strict black 
        and white using a threshold of 200.
        """
        threshold = 200
        binary_image = np.zeros_like(image)
        binary_image[image > threshold] = 255
        return binary_image
    
    def line_distance(self, image):
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
        VERTICAL_PADDING_RATIO = 0.03  # 3% of line height as padding
        MIN_LINE_HEIGHT = 20           # Minimum height to consider a valid line

        line_boundaries = []
        line_ys = sorted([int((float(y0) + float(y1))/2) for (y0,y1) in self.line_coords])
        
        # If no lines detected we use the whole image as one block.
        if not line_ys:
            return [[0, logo.shape[0], 0, logo.shape[1]]]

        half_line_height = int(self.line_height * 0.5)

        # Creating one box per line.
        for i in range(0, len(line_ys), 2):
            # Top boundary (current line position + padding).
            padding = int(self.line_height * VERTICAL_PADDING_RATIO)
            top = max(0, line_ys[i] - padding)
            
            # Bottom boundary (next line or end of image).
            if i < len(line_ys) - 1:
                bottom = line_ys[i+1] + padding + half_line_height
            else:
                bottom = logo.shape[0]
            
            # Skipping if the line height is too small.
            if (bottom - top) < MIN_LINE_HEIGHT:
                continue
                
            line_boundaries.append([
                top,           # y1
                bottom,        # y2
                0,             # x1 (start at left edge)
                logo.shape[1]  # x2 (end at right edge)
            ])

        # Visualization.
        color_logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR)
        
        # Drawing notebook lines (from the original image with lines).
        for line_y in line_ys:
            cv2.line(color_logo, (0, line_y), (color_logo.shape[1], line_y), 
                    (255, 0, 0), 1)
        
        # Drawing text blocks.
        for i, (y1, y2, x1, x2) in enumerate(line_boundaries):
            cv2.rectangle(color_logo, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(color_logo, f"Block {i+1}", (x1 + 10, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(color_logo, cv2.COLOR_BGR2RGB))
        plt.title("Full-width Line Boxes")
        plt.axis('off')
        plt.show()
        plt.imsave("BLOCKS.png", color_logo)

        return line_boundaries
    
    def detect_spaces(self, logo, xycoords):
        """
        Uses k-means clustering to detect spaces between 
        words based on vertical whitespace columns.
        """
        spaces = [0]
        SPACE_SENSITIVITY = 0.5       # Lower = more sensitive to small spaces (0.5-1.0).
        MIN_SPACE_WIDTH = 100         # Absolute minimum pixels to consider as space.
        WORD_SPACE_WIDTH_RATIO = 0.4  # Ratio of line_height for word separation.
        MIN_GAP_RATIO = 0.5           # Minimum ratio of line_height to be considered.

        for y_start, y_end, x_start, x_end in xycoords:
            segment = self.black_and_white(logo[y_start:y_end, x_start:x_end])
            _, width = segment.shape
            whitespace_lengths = []
            white_count = 0

            for col in range(width):
                column = segment[:, col]
                if np.any(column == 0): # Contains text. 
                    if white_count > MIN_SPACE_WIDTH:
                        whitespace_lengths.append(white_count)
                    white_count = 0
                else: # Whitespace.
                    white_count += 1

            if white_count > MIN_SPACE_WIDTH:
                whitespace_lengths.append(white_count)

            # Classifying spaces using clustering.
            non_zero_spaces = np.array([w for w in whitespace_lengths if w > MIN_SPACE_WIDTH])

            if non_zero_spaces.size == 0:
                spaces.append(2) # Newline.
            else:
                if len(non_zero_spaces) > 1:
                    # Using percentiles.
                    q25 = np.percentile(non_zero_spaces, 25)
                    q75 = np.percentile(non_zero_spaces, 75)
                    threshold = q25 + SPACE_SENSITIVITY * (q75 - q25)
                    
                    # Classifying each space.
                    for space in whitespace_lengths:
                        if space >= threshold:
                            spaces.append(2)  # Word space.
                        elif space >= MIN_SPACE_WIDTH:
                            spaces.append(1)  # Character space.
                else:
                    # Fallback for lines with only one space
                    spaces.extend([2 if s > MIN_SPACE_WIDTH+1 else 1 
                                for s in whitespace_lengths])
                
                spaces.append(2)  # End of line marker.

        return np.array(spaces)
    
    def segment_words(self, image, finalXY):
        """
        Segments words from line segments using space 
        classification (1=intra-word, 2=inter-word) and 
        saves each word as a separate PNG file.
        """
        self.char_positions = []
        MIN_WORD_WIDTH = 5            # Minimum width to consider a word segment.
        WORD_SPACE_WIDTH_RATIO = 0.4  # Ratio of line_height for word separation.
        MIN_GAP_RATIO = 0.5           # Minimum ratio of line_height to be considered.

        if self.line_height > 0:
            word_separation_threshold = self.line_height * WORD_SPACE_WIDTH_RATIO
            min_whitespace_pixels = self.line_height * MIN_GAP_RATIO
        else:
            word_separation_threshold = 20  
            min_whitespace_pixels = 2 

        # Ensuring min_whitespace_pixels is reasonably 
        # smaller than word_separation_threshold.
        if (min_whitespace_pixels >= word_separation_threshold) and \
           (self.line_height <= 0):
            min_whitespace_pixels = max(1, word_separation_threshold - 1)
        elif (min_whitespace_pixels >= word_separation_threshold) and \
             (self.line_height > 0):
            min_whitespace_pixels = max(1, self.line_height * (WORD_SPACE_WIDTH_RATIO / 2) )

        for line_idx, (top, bottom, left, right) in enumerate(finalXY):
            line_region = image[top:bottom, left:right]
            
            column_pos = 0

            # Scanning to find all space segments and their types.
            segment = self.black_and_white(line_region)
            _, width = segment.shape
            whitespace_lengths = []
            white_count = 0

            for col in range(width):
                column = segment[:, col]
                if np.any(column == 0):  # Contains text.
                    if white_count >= min_whitespace_pixels:
                        whitespace_lengths.append((white_count, column_pos - white_count))
                        white_count = 0
                else:  # Whitespace.
                    white_count += 1
                column_pos += 1

            if white_count > min_whitespace_pixels:
                whitespace_lengths.append((white_count, column_pos - white_count))
            
            classified_spaces = []
            for length, pos in whitespace_lengths:
                if length > 0:
                    space_type = 2 if length >= word_separation_threshold else 1
                    classified_spaces.append((pos, pos + length, space_type))
                else:
                    classified_spaces.append((pos, pos + length, 0))
            
            # Splitting the line at type 2 spaces.
            word_start = 0
            word_count = 0
            
            for space in classified_spaces:
                space_start, space_end, space_type = space
                
                if space_type == 2:  # Word boundary found.
                    word_end = space_start
                    if word_end - word_start >= MIN_WORD_WIDTH:
                        # Extracting and saving the word.
                        word_region = line_region[:, word_start:word_end]
                        
                        # Finding vertical bounds.
                        rows_with_ink = np.any(word_region < 200, axis=1)
                        y_positions = np.where(rows_with_ink)[0]
                        
                        abs_x1 = word_start
                        abs_x2 = word_end
                        abs_y1 = top + (y_positions.min() if len(y_positions) > 0 else 0)
                        abs_y2 = top + (y_positions.max() + 1 if len(y_positions) > 0 else (bottom - top))

                        filename = f"line{line_idx}_word{word_count}.png"
                        self.letter_coord['letters'][filename] = [
                            [int(abs_x1), int(abs_x2)],
                            [int(abs_y1), int(abs_y2)]
                        ]
                        
                        bw_word = self.black_and_white(word_region).astype(np.uint8)
                        Image.fromarray(bw_word).save(f"{DUMP}/{filename}")
                        word_count += 1
                    
                    word_start = space_end  # Starting the next word after this space.
            
            # Save the last word in the line
            if width - word_start >= MIN_WORD_WIDTH:
                word_region = line_region[:, word_start:width]
                
                rows_with_ink = np.any(word_region < 200, axis=1)
                y_positions = np.where(rows_with_ink)[0]
                
                abs_x1 = word_start
                abs_x2 = word_start + width 
                abs_y1 = top + (y_positions.min() if len(y_positions) > 0 else 0)
                abs_y2 = top + (y_positions.max() + 1 if len(y_positions) > 0 else (bottom - top))

                filename = f"line{line_idx}_word{word_count}.png"
                self.letter_coord['letters'][filename] = [
                    [int(abs_x1), int(abs_x2)],
                    [int(abs_y1), int(abs_y2)]
                ]
                
                bw_word = self.black_and_white(word_region).astype(np.uint8)
                Image.fromarray(bw_word).save(f"{DUMP}/{filename}")

        return
    
    def clean_and_resize(self, path):
        """
        Processes character images by removing borders and 
        standardizing size for OCR. The x border is as tight
        as possible and the y border extends upwards until
        a few pixels above the line above the letter to cover
        tonos cases and to match the data the model was trained on.
        """
        def has_black_pixels(column_or_row):
            """Check if the column/row contains any black pixels (0)."""
            return any(pixel == 0 for pixel in column_or_row)
        
        img_array = io.imread(path)
        filename = os.path.basename(path)
        
        # Get original coordinates before cleaning
        if filename in self.letter_coord['letters']:
            orig_coords = self.letter_coord['letters'][filename]
            orig_x1, orig_x2 = orig_coords[0]
            orig_y1, orig_y2 = orig_coords[1]
            
            # Find content bounds in x-axis
            x1, x2 = 0, img_array.shape[1]
            for i in range(img_array.shape[1]):
                if has_black_pixels(img_array[:, i]):
                    x1 = max(0, i-1)
                    break
                    
            for i in range(img_array.shape[1]-1, -1, -1):
                if has_black_pixels(img_array[:, i]):
                    x2 = min(img_array.shape[1], i+3)
                    break
            
            # Calculating new absolute x coordinates.
            new_x1 = orig_x1 + x1
            new_x2 = min(orig_x2, orig_x1 + x2)
            
            # Finding actual vertical bounds after cleaning.
            rows_with_ink = np.any(img_array < 200, axis=1)
            y_positions = np.where(rows_with_ink)[0]

            #padding = 0.04 * self.line_height
            if len(y_positions) > 0:
                rel_y1 = y_positions.min()
                rel_y2 = y_positions.max() + 1 
                
                new_y1 = orig_y1 + rel_y1
                new_y2 = orig_y1 + rel_y2
            else:
                new_y1, new_y2 = orig_y1, orig_y2
            
            self.letter_coord['letters'][filename] = [
                [new_x1, new_x2],
                [int(new_y1), int(new_y2)]
            ]

        # Cropping the image.
        cropped_img = img_array[:, x1:x2]

        # Converting to black and white and saving.
        final_array = self.black_and_white(cropped_img)
        Image.fromarray(final_array).save(path)

    def crop_original_img(self):
        '''
        Takes word coordinates and crops the 
        original image to send it to the model 
        for the final classification.
        '''
        image = Image.open(IMG_PATH)

        for filename, coords in self.letter_coord['letters'].items():
            x1, x2 = coords[0]
            y1, y2 = coords[1]

            cropped = image.crop((x1, y1, x2, y2))

            output_path = os.path.join(OUTPUT_FOLDER, filename)
            cropped.save(output_path)

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
        returns the recognized text with statistics. The
        input image loading, text blocks, space and average
        height between each notebook line are initialized 
        in __init__ to avoid clutter.
        """
        # Segments words and saves them as PNGs in LetterDump.
        self.segment_words(self.img_no_lines, self.block_coords)
        print("Character segmentation completed.")
        
        # Processes each word. (tightly resizes the x axis)
        for im in os.listdir(DUMP+"/"):
            self.clean_and_resize(DUMP+"/"+str(im))

        # Printing the coordinates of each word found.
        for filename, coords in recognizer.letter_coord['letters'].items():
            print(f"{filename}: x={coords[0]}, y={coords[1]}")

        # Using the coordinates of each letter to crop the original image
        # without cv2 transforms. WIP, might not be needed. 
        # TODO: when making the boxes in segment_words, crop the original
        # image as well with the coords obtained.
        #self.crop_original_img()
        
        # Recognizing each character added to the image folder.
        charPred = []
        char_data = []
        for i in sorted(os.listdir(DUMP+"/")):
            img_path = os.path.join(DUMP, i)
            with Image.open(img_path) as image:
                char, conf = self.recognize_character(np.asarray(image))
                char_data.append((char, conf))
            #os.remove(img_path)
        
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

# ----------------------------------------------------------------------------#
