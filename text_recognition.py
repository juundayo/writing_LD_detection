# ----------------------------------------------------------------------------#

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
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
        plt.show()

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
        plt.show()

        return binary

    def black_and_white(self, a):
        """
        Converts grayscale image to strict black and white
        using a threshold of 200.
        """
        m = a.copy()
        for i in range(len(m)):
            for j in range(len(m[0])):
                if m[i][j] > 200:
                    m[i][j] = 255  # Background.
                else:
                    m[i][j] = 0    # Letter.
        return m
    
    def find_line_coordinates(self, logo):
        """
        Detects lines of text in the image and returns their
        bounding coordinates.
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
        return xycoords
    
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
        Segments individual characters from line segments and stores
        their x y coordinates in a dictionary fromatted as followed:
        1: [coordinates], 2:[coordinates], ... , n: [coordinates]
        where n indicates the 2 lines above and below each character.
        For example, an input with 3 lines will use 2 keys in the
        dictionary. 

        After each character coordinate is found, we edit the height
        boundary of each character so that it extends until the line
        of the notebook above the character that was found. (We do
        that to match the way that the model's training dataset was
        created).
        """
        col = []
        count = 0
        
        for hoe in finalXY:
            newC = [0] # Starting with leftmost boundary.
            
            def flagCalc(i):
                flag = 1
                for j in range(len(logo[hoe[0]:hoe[1],hoe[2]:hoe[3]][:,i])):
                    if logo[hoe[0]:hoe[1],hoe[2]:hoe[3]][:,i][j] < 150:
                        flag = 0 # Pixel is part of the character (dark).
                return flag
            
            # Find character boundaries. (start, end)
            for i in range(len(logo[hoe[0]:hoe[1],hoe[2]:hoe[3]][0,:])):
                try:
                    if flagCalc(i) < flagCalc(i+1):
                        newC.append(i+1) 
                except:
                    pass
            
            newC.append(hoe[3]) # Adds rightmost boundary.
            col.append(newC) # Saving x coordinates for the line.
            
            # Saving each character.
            for i in range(len(newC)-1):
                A = self.black_and_white(logo[hoe[0]:hoe[1],hoe[2]:hoe[3]][:,newC[i]:newC[i+1]])
                A = A.astype(np.uint8)
                im = Image.fromarray(A)
                im.save(DUMP+"/"+str(count)+".png")
                count += 1
        return col
    
    def clean_and_resize(self, path):
        """
        Processes character images by removing borders and
        standardizing size for OCR.
        """
        a = io.imread(path)

        def flagCalc(i):
            flag = 0
            for j in range(len(i)):
                if i[j] == 0:
                    flag = 1
            return flag
        
        # Find content boundaries.
        y1 = 0
        y2 = a.shape[0]
        x1 = 0
        x2 = a.shape[1]
        
        for i in range(len(a)-1):
            if flagCalc(a[i]) < flagCalc(a[i+1]):
                y2 = a.shape[0]
                if (i+1) < y2:
                    y1 = i+1
            elif flagCalc(a[i]) > flagCalc(a[i+1]):
                if (i-1) > y1:
                    y2 = i-1
        
        for i in range(len(a[0,:])-1):
            if flagCalc(a[:,i]) < flagCalc(a[:,i+1]):
                if (i+1) < x2:
                    x1 = i+1
            elif flagCalc(a[:,i]) > flagCalc(a[:,i+1]):
                if (i-1) > x1:
                    x2 = i-1
        
        # Crop and save
        im = Image.fromarray(a[y1:y2,x1:x2])
        im.save(path)
        
        # Resize with padding
        a = io.imread(path)
        if a.shape[0] > a.shape[1]:
            f = 28/a.shape[0]
        else:
            f = 28/a.shape[1]
        
        b = Image.fromarray(a, mode='L').resize((
            int(a.shape[1]*f), int(a.shape[0]*f)), Image.BICUBIC)
        
        c = Image.fromarray(np.full((32, 32), 255).astype('uint8'), mode='L')

        img_w, img_h = b.size
        bg_w, bg_h = c.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)

        c.paste(b, offset)
        c.save(path)
        
        # Final thresholding
        a_bw = io.imread(path)
        a_bw = self.black_and_white(a_bw)

        Image.fromarray(a_bw).save(path)
    
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
        returns the recognized text with statistics.
        """
        # Processes the input image with no lines
        # and detects the line coordinates.
        logo = self.input_without_lines(IMG_PATH)
        xycoords = self.find_line_coordinates(logo)

        # Detects spaces.
        spaces = self.detect_spaces(logo, xycoords)
        
        # Segments characters.
        self.segment_characters(logo, xycoords)
        print("Character segmentation completed.")
        
        # Processes each character.
        for mm in os.listdir(DUMP+"/"):
            self.clean_and_resize(DUMP+"/"+str(mm))
        
        # Recognizing each character added to the image folder.
        co = -1
        charPred = []
        confidence_scores = []
        
        while True:
            co += 1
            try:
                image = Image.open(DUMP+"/"+str(co)+'.png')
                char_array = np.asarray(image)
                char, conf = self.recognize_character(char_array)
                
                charPred.append(char)
                confidence_scores.append(conf)
                # Keeping this commented for now. 
                #os.remove(DUMP+"/"+str(co)+'.png')
            except:
                break
        
        # Reconstructing the text with dictionary checking.
        result = self.reconstruct_text(charPred, spaces)
        
        return result

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
