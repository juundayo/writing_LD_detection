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
from skimage import io
from collections import defaultdict

from vt_model import OCR
from data_loading import class_mapping

# ----------------------------------------------------------------------------#

MODEL_PATH = "/home/ml3/Desktop/Thesis/rs1_tt2_v2.pth"
IMG_PATH = "/home/ml3/Desktop/Thesis/.venv/model_test.jpg"
DUMP = "/home/ml3/Desktop/Thesis/LetterDump"
THRESHOLD = 100

# ----------------------------------------------------------------------------#

class GreekTextRecognizer:
    def __init__(self, model_path, device):
        # Creating the generated cropped image folder. 
        os.makedirs(DUMP, exist_ok=True)

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

        # Calculating the coordinates of each line of the notebook.
        self.line_coords = self.find_line_coordinates(self.img_no_lines)

        # Detects space boxes for characters.
        self.spaces = self.detect_spaces(self.img_no_lines, self.line_coords)

        # The space between the lines.
        self.line_space = self.line_distance(self.img_lines)
        
        # Transform applied to each generated cropped image.
        self.transform = transforms.Compose([
            transforms.Resize((516, 78)),
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
        threshold = THRESHOLD
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
    
    def find_line_coordinates(self, logo):
        """
        Detects lines of text in the image and 
        returns their bounding coordinates.
        """
        coords = []
        xycoords = []
        
        def line_coords(coords):
            xmin = coords[0][0][0]
            xmax = coords[-1][0][0]
            ymin = 20000
            ymax = 0
            
            for i in coords:
                for j in i:
                    if j[1] > ymax:
                        ymax = j[1]
                    if j[1] < ymin:
                        ymin = j[1]
            
            xycoords.append([xmin, xmax+2, ymin+1, ymax])
        
        for i in range(len(logo[1:-1,1:-1])):
            coo = []
            flag = 0
            
            # Check if line contains text.
            for c in logo[1:-1,1:-1][i]:
                if c < 200:
                    flag = 1
            
            if flag == 1:
                # Find character boundaries.
                for b in range(len(logo[1:-1,1:-1][i])):
                    if logo[1:-1,1:-1][i][b] > 200:
                        try:
                            if logo[1:-1,1:-1][i][b+1] < 200:
                                coo.append([i,b+1])
                        except:
                            pass
                    
                    if logo[1:-1,1:-1][i][b] < 200:
                        try:
                            if logo[1:-1,1:-1][i][b+1] > 200:
                                coo.append([i,b])
                        except:
                            pass
            else:
                if len(coords) > 0:
                    line_coords(coords)
                    coords = []
            
            if len(coo) > 0:
                coords.append(coo)
        
        print("Line Coordinates Found:", xycoords)

        color_logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR)

        for i, (x1, x2, y1, y2) in enumerate(xycoords):
            cv2.rectangle(color_logo, (y1, x1), (y2, x2), (0, 255, 0), 1)
            label = f"Line {i+1}: [{y1},{x1}]→[{y2},{x2}]"
            cv2.putText(color_logo, label, (y1, x1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(color_logo, cv2.COLOR_BGR2RGB))
        plt.title("Detected Notebook Lines with Bounding Boxes")
        plt.axis('off')
        #plt.show()

        return xycoords
    
    def line_distance(self, image, show=True, save=False):
        '''
        Dynamically detecting the lines of the input image using Hough
        transform to calculating the distance between each line. 
        '''
        inverted = ~image
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
        hspace, theta, dist = hough_line(inverted, tested_angles)
        h, q, d = hough_line_peaks(hspace, theta, dist)

        angle_list = []
        dist_list = []

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
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

        origin = np.array((0, inverted.shape[1]))

        # Sorting the lines by their distance (y-coordinate).
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

        ax[2].set_xlim(origin)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        plt.tight_layout()
        plt.show()

        print(f"Average distance between lines: {avg_line_distance} pixels")
        return avg_line_distance
    
    def detect_spaces(self, logo, xycoords):
        """
        Uses k-means clustering to detect spaces between words
        based on vertical whitespace columns.
        """
        spaces = np.array([0])
        ctr = 0
        
        for y in xycoords:
            sp = []
            a = self.black_and_white(logo[y[0]:y[1],y[2]:y[3]])
            
            # Count consecutive white columns
            for i in range(len(a[0,:])):
                f = 0
                for j in a[:,i]:
                    if j == 0:
                        f = 1
                
                if f != 1:
                    ctr += 1
                if f == 1:
                    sp.append(ctr)
                    ctr = 0
            
            nums = np.array([jj for jj in sp if jj != 0])
            
            if len(nums) == 0:
                spaces = np.concatenate((spaces, np.array([2])), axis=None)
            else:
                # Cluster space widths into two groups (small and large spaces).
                km = ckwrap.ckmeans(nums, 2)
                spaces = np.concatenate((spaces, km.labels, np.array([2])), axis=None)
        
        return spaces
    
    def segment_characters(self, logo, finalXY):
        """
        Segments individual characters from line segments and
        saves them as a png that will later be processed by  
        the clean_and_resize function.
        """
        all_columns = []
        count = 0
        self.char_positions = []

        for top, bottom, left, right in finalXY:
            region = logo[top:bottom, left:right]
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

            for i in range(len(boundaries) - 1):
                char_region = region[:, boundaries[i]:boundaries[i + 1]]
                bw_char = self.black_and_white(char_region).astype(np.uint8)
                Image.fromarray(bw_char).save(f"{DUMP}/{count}.png")
                count += 1

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
        
        # Loading the letter and finding content boundaries.
        img_array = io.imread(path)
        height, width = img_array.shape

        # Getting the character's position in the original image.
        # The filename contains the character's index in the sequence.
        char_index = int(os.path.basename(path).split('.')[0])

        # Finding which line this character belongs to.
        current_line = None
        for line_idx, line in enumerate(self.line_coords):
            # Check if character is between line start and end.
            if line[0] <= char_index <= line[1]:
                current_line = line
                break
        
        # Finding vertical boundaries (y1, y2).
        y1, y2 = 0, height

        # Finding the upper boundary (go up to the line above).
        if current_line is not None:
            # Get y-coordinate of the line above (current_line[2] is ymin of the line).
            # We'll go up to 5 pixels above the line.
            y1 = max(0, current_line[2] - 5)  

        # Finding the bottom boundary (bottom of the character).
        for i in range(height - 1):
            current_row_has_black = has_black_pixels(img_array[i])
            next_row_has_black = has_black_pixels(img_array[i + 1])
            
            if current_row_has_black and not next_row_has_black:
                y2 = i
        
        # Finding horizontal boundaries (x1, x2).
        x1, x2 = 0, width
        for i in range(width - 1):
            current_col_has_black = has_black_pixels(img_array[:, i])
            next_col_has_black = has_black_pixels(img_array[:, i + 1])
            
            if not current_col_has_black and next_col_has_black:
                x1 = i + 1
            elif current_col_has_black and not next_col_has_black:
                x2 = i
        
        # Cropping the image.
        cropped_img = Image.fromarray(img_array[y1:y2, x1:x2])
        cropped_img.save(path)
        
        # Resizing with aspect ratio preservation.
        img_array = io.imread(path)
        max_dim = max(img_array.shape)
        scale_factor = 28 / max_dim
        new_size = (int(img_array.shape[1] * scale_factor), 
                    int(img_array.shape[0] * scale_factor))
        
        resized_img = Image.fromarray(img_array, mode='L').resize(
            new_size, Image.BICUBIC)
        
        # Centering on 32x32 white background.
        background = Image.new('L', (32, 32), color=255)
        offset = ((32 - resized_img.width) // 2, 
                (32 - resized_img.height) // 2)
        background.paste(resized_img, offset)
        
        # Applying final thresholding and saving.
        final_img = self.black_and_white(np.array(background))
        Image.fromarray(final_img).save(path)
    
    def recognize_character(self, char_image):
        """
        Recognizing a single character.
        Returns the character itself and the 
        respective confidence of the prediction.
        """
        # Original image.
        plt.figure(figsize=(10, 4))
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
        Reconstructs the recognized characters into words and lines,
        using the space information and dictionary lookup.
        """
        full_text = []
        current_word = []
        current_confidences = []
        confidence_scores = []
        word_stats = defaultdict(int)
        
        co = 0
        for char in charPred:
            if spaces[co] == 1:  # Space between words.
                if current_word:
                    word = ''.join(current_word)
                    avg_conf = sum(current_confidences)/len(current_confidences) if current_confidences else 0
                    marked_word = self.mark_unknown(word, avg_conf)
                    full_text.append(marked_word)
                    current_word = []
                    current_confidences = []
                full_text.append(' ')
            elif spaces[co] == 2:  # New line.
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
        input image loading, notebook line and space finding
        is initialized in __init__.
        """
        # Segments characters.
        self.segment_characters(self.img_no_lines, self.line_coords)
        print("Character segmentation completed.")
        
        # Processes each character.
        for im in os.listdir(DUMP+"/"):
            self.clean_and_resize(DUMP+"/"+str(im))
        
        # Recognizing each character added to the image folder.
        co = -1
        charPred = []
        confidence_scores = []
        
        char_data = []
        for i in os.listdir(DUMP+"/"):
            img_path = os.path.join(DUMP, f"{i}.png")
            with Image.open(img_path) as image:
                char, conf = self.recognize_character(np.asarray(image))
                char_data.append((char, conf))
            os.remove(img_path)
        
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
