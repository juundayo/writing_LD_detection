# ----------------------------------------------------------------------------#

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.transform import (hough_line, hough_line_peaks)
from PIL import Image, ImageOps
import ckwrap
import shutil
from skimage import io
from collections import defaultdict

from vt_model import OCR
from data_loading import class_mapping

# ----------------------------------------------------------------------------#

MODEL_PATH = "/home/ml3/Desktop/Thesis/rs1_tt2_v2.pth"
IMG_PATH = "/home/ml3/Desktop/Thesis/.venv/model_test.jpg"
#IMG_PATH = "/home/ml3/Desktop/model_test2.JPG"
DUMP = "/home/ml3/Desktop/Thesis/LetterDump"
OUTPUT_FOLDER = "/home/ml3/Desktop/Thesis/LetterCrops"
IMG_HEIGHT = 512
IMG_WIDTH = 78

# ----------------------------------------------------------------------------#

class GreekTextRecognizer:
    def __init__(self, model_path, device):
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

        # Loading the model.
        self.device = device
        self.model = self.load_model(model_path, len(class_mapping))

        # Loading the dictionary.
        self.greek_dictionary = self.load_greek_dictionary(
            '/home/ml3/Desktop/Thesis/.venv/Data/filtered_dictionary.dic'
        ) 

        # Inverse mapping from index to class name.
        self.classes = sorted(class_mapping.values())
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        # Loading edited versions of the input image. 
        self.img_lines = self.input_with_lines(IMG_PATH)
        self.img_no_lines = self.input_without_lines(IMG_PATH)

        # Calculating the space between the notebook lines.
        self.line_height, self.line_coords = self.line_distance(self.img_lines)

        # Calculating the coordinates of blocks of text recognized.
        self.block_coords = self.find_blocks(self.img_no_lines)

        # Detecting space boxes for characters.
        self.spaces = self.detect_spaces(self.img_no_lines, self.block_coords)
        
        # Transform applied to each generated cropped image.
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def load_model(self, model_path, num_classes):
        model = OCR(num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
   
        print("Loaded the model!")
        return model
    
    def load_greek_dictionary(self, dict_path):
        """Loading the Greek dictionary into a set for faster lookup."""
        dictionary = set()

        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    dictionary.add(word)

        print('Loaded the dictionary!')
        return dictionary
    
    def mark_unknown(self, word, confidence=0):
        if (word.lower() not in self.greek_dictionary) or (confidence < 0.7):
            return f"*{word}*"
        
        return word
        
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
        image = 255 - image # Inveting black and white.

        # Show the image
        plt.imshow(image, cmap='gray')
        plt.title("Thresholded Binary Image")
        plt.axis('off')
        #plt.show()

        return image
    
    def input_without_lines(self, IMAGE_PATH):
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

        return binary

    def black_and_white(self, image):
        """
        Converts grayscale image to strict black and white
        using a threshold of 200.
        """
        threshold = 200
        binary_image = np.zeros_like(image)
        binary_image[image > threshold] = 255
        return binary_image
    
    def line_distance(self, image):
        '''
        Dynamically detecting the lines of the input image using Hough
        transform to calculating the distance between each line. 
        '''
        angle_list = []
        dist_list = []
        line_coords = []

        inverted = ~image
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
        hspace, theta, dist = hough_line(inverted, tested_angles)
        _, q, d = hough_line_peaks(hspace, theta, dist)

        _, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()
        ax[0].imshow(inverted, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()
        ax[1].imshow(np.log(1 + hspace),
                extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), 
                        dist[-1], dist[0]], cmap='gray', aspect=1/1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')
        ax[2].imshow(inverted, cmap='gray')

        # Sorting the lines by their distance (y-coordinate).
        origin = np.array((0, inverted.shape[1]))
        sorted_indices = np.argsort(d)
        sorted_distances = d[sorted_indices]
        sorted_angles = q[sorted_indices]

        line_distances = []
        for i in range(1, len(sorted_distances)):
            line_distances.append(sorted_distances[i] - sorted_distances[i-1])

        # Calculating the average distance between lines.
        avg_line_distance = np.mean(line_distances) if line_distances else 0

        for angle, dist in zip(sorted_angles, sorted_distances):
            angle_list.append(angle)
            dist_list.append(dist)
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[2].plot(origin, (y0, y1), '-r')
            line_coords.append((y0, y1))

        ax[2].set_xlim(origin)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')
        plt.tight_layout()
        plt.show()

        # Storing the line coordinates for later use.
        self.detected_lines = sorted_distances

        print(f"Average distance between lines: {avg_line_distance} pixels")
        print(f"Line coordinates:", line_coords)
        return avg_line_distance, line_coords
    
    def find_blocks(self, logo):
        """
        Detects lines of text in the image and returns
        their bounding coordinates. We use a threshold
        of width >= 5 to remove noise. (Dots, etc.)
        """
        line_boundaries = []
        MIN_BLOCK_WIDTH = 5
        
        def calculate_line_boundary(coords_list):
            """
            Calculating the bounding box for a 
            complete line from character coordinates.
            """
            x_coords = [coord[0][0] for coord in coords_list]
            y_coords = [j[1] for coord in coords_list for j in coord]
            
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            
            # Adding a small padding to the boundaries.
            padding = 3
            return [xmin, xmax + padding, max(0, ymin - padding), ymax + padding]

        # Check 1 -> Detecting all potential character transitions.
        transitions = []
        for row_idx, row in enumerate(logo[1:-1, 1:-1]):
            for col_idx in range(len(row) - 1):
                current_pixel = row[col_idx]
                next_pixel = row[col_idx + 1]
                
                # Transition points (background<->text).
                if (current_pixel > 200 and next_pixel < 200) or \
                (current_pixel < 200 and next_pixel > 200):
                    transitions.append((row_idx + 1, col_idx + 1))  # +1 for border offset.
        
        # Check 2 -> Group transitions into lines with tonos awareness.
        if transitions:
            # Sorting transitions by row then column.
            transitions.sort()
            
            current_row = transitions[0][0]
            line_segments = []
            
            for row_idx, col_idx in transitions:
                # Checking if we've moved to a new line.
                if abs(row_idx - current_row) > self.line_height * 1.1:
                    if line_segments:
                        boundary = calculate_line_boundary(line_segments)
                        block_width = boundary[3] - boundary[2] # ymax - ymin.

                        if block_width >= MIN_BLOCK_WIDTH:
                            line_boundaries.append(boundary)

                        line_segments = []
                    current_row = row_idx
                
                # Checking if this is likely a tonos (small vertical displacement).
                is_tonos = False
                if line_segments:
                    last_col = line_segments[-1][-1][1]
                    if abs(col_idx - last_col) < 10:  # Tonos appears close to previous character.
                        is_tonos = True
                
                if not is_tonos:
                    line_segments.append([(row_idx, col_idx)])
            
            # Adding the last line.
            if line_segments:
                boundary = calculate_line_boundary(line_segments)
                block_width = boundary[3] - boundary[2]
                if block_width >= MIN_BLOCK_WIDTH:
                    line_boundaries.append(boundary)
                
        print("Line block coordinates:", line_boundaries)

        color_logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR)

        for i, (x1, x2, y1, y2) in enumerate(line_boundaries):
            cv2.rectangle(color_logo, (y1, x1), (y2, x2), (0, 255, 0), 1)
            label = f"Line {i+1}: [{y1},{x1}]→[{y2},{x2}]"
            cv2.putText(color_logo, label, (y1, x1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(color_logo, cv2.COLOR_BGR2RGB))
        plt.title("Detected Notebook Lines with Bounding Boxes")
        plt.axis('off')
        plt.show()

        return line_boundaries
    
    
    def detect_spaces(self, logo, xycoords):
        """
        Uses k-means clustering to detect spaces between words
        based on vertical whitespace columns.
        """
        spaces = [0]

        for y_start, y_end, x_start, x_end in xycoords:
            segment = self.black_and_white(logo[y_start:y_end, x_start:x_end])
            _, width = segment.shape
            whitespace_lengths = []
            white_count = 0

            for col in range(width):
                column = segment[:, col]
                if np.any(column == 0):  # Contains a black pixel
                    if white_count > 0:
                        whitespace_lengths.append(white_count)
                        white_count = 0
                else:
                    white_count += 1

            if white_count > 0:
                whitespace_lengths.append(white_count)

            non_zero_spaces = np.array([w for w in whitespace_lengths if w > 0])

            if non_zero_spaces.size == 0:
                spaces.append(2)
            else:
                # Cluster into small/large spaces
                labels = ckwrap.ckmeans(non_zero_spaces, 2).labels
                spaces.extend(labels)
                spaces.append(2)

        return np.array(spaces)
    
    def segment_characters(self, image, finalXY):
        """
        Segments individual characters from line segments and
        saves them as a png that will later be processed by  
        the clean_and_resize function.
        """
        all_columns = []
        self.char_positions = []
        MIN_CHAR_WIDTH = 5

        for line_idx, (top, bottom, left, right) in enumerate(finalXY):
            region = image[top:bottom, left:right]
            width = region.shape[1]

            def is_background_column(i):
                return all(pixel >= 150 for pixel in region[:, i])

            boundaries = [0]
            try:
                boundaries += [i + 1 for i in range(width - 1)
                            if is_background_column(i) and not is_background_column(i + 1)]
            except IndexError:
                pass

            boundaries.append(width)
            all_columns.append(boundaries)

            char_count = 0
            for i in range(len(boundaries) - 1):
                char_width = boundaries[i+1] - boundaries[i]
                if char_width >= MIN_CHAR_WIDTH:
                    char_region = region[:, boundaries[i]:boundaries[i + 1]]
                    
                    # Finding the vertical bounds of the character.
                    rows_with_ink = np.any(char_region < 200, axis=1)
                    y_positions = np.where(rows_with_ink)[0]
                    
                    abs_x1 = left + boundaries[i]
                    abs_x2 = left + boundaries[i + 1]
                    abs_y1 = top + (y_positions.min() if len(y_positions) > 0 else 0)
                    abs_y2 = top + (y_positions.max() + 1 if len(y_positions) > 0 else (bottom - top))

                    filename = f"{line_idx}_{char_count}.png"
                    # Store absolute coordinates with proper height
                    self.letter_coord['letters'][filename] = [
                        [int(abs_x1), int(abs_x2)],  # absolute x coordinates
                        [int(abs_y1), int(abs_y2)]   # absolute y coordinates
                    ]
                    
                    bw_char = self.black_and_white(char_region).astype(np.uint8)
                    
                    Image.fromarray(bw_char).save(f"{DUMP}/{filename}")
                    char_count += 1

        return all_columns
    
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
                    x2 = min(img_array.shape[1], i+2)
                    break
            
            # Calculating new absolute x coordinates.
            new_x1 = orig_x1 + x1
            new_x2 = orig_x1 + x2
            
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

        # Transform and prepare for model
        char_pil = Image.fromarray(char_image).convert('L')
        transformed_img = self.transform(char_pil)
        
        # Prepare transformed image for display
        img_to_show = transformed_img.squeeze().cpu().numpy()
        img_to_show = (img_to_show * 0.5) + 0.5  # Undo normalization
        img_to_show = np.clip(img_to_show, 0, 1)
        
        # Transformed image
        plt.subplot(1, 2, 2)
        plt.imshow(img_to_show, cmap='gray')
        plt.title("Transformed (Model Input)")
        plt.axis('off')
        plt.style.use('dark_background')

        plt.tight_layout()
        plt.show()
        
        # Continue with recognition
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
        # Segments characters.
        self.segment_characters(self.img_no_lines, self.block_coords)
        print("Character segmentation completed.")
        
        # Processes each character.
        for im in os.listdir(DUMP+"/"):
            self.clean_and_resize(DUMP+"/"+str(im))

        # Printing the coordinates of each letter found.
        for filename, coords in recognizer.letter_coord['letters'].items():
            print(f"{filename}: x={coords[0]}, y={coords[1]}")

        self.crop_original_img()
        
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
    recognizer = GreekTextRecognizer(MODEL_PATH, "cuda")
    result = recognizer.recognize_text()
    
    print("\nFinal Results:")
    print(f"Recognized Text: {result['text']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Character Count: {result['char_count']}")
    print(f"Word Count: {result['word_count']}")
    print(f"Unknown Words: {result['unknown_words']}")

# ----------------------------------------------------------------------------#
