# ----------------------------------------------------------------------------#

import cv2
import torch
from torchvision import transforms
from PIL import Image

from main import LetterOCR
from data_loading import class_mapping

# ----------------------------------------------------------------------------#

class GreekTextRecognizer:
    def __init__(self, model_path, device):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = LetterOCR(num_classes=len(class_mapping)).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Inverse mapping from index to class name.
        self.idx_to_class = {v: k for k, v in class_mapping.items()}
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.SPACE_CLASS = 'space'
        self.HYPHEN_CLASS = 'hyphen'
    
    def preprocess_image(self, image_path):
        """Loads and preprocesses the image for character detection."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image

    def line_grouping(self, characters):
        '''Groups characters into lines based on their y-coordinate.'''
        sorted_chars = sorted(characters, key=lambda c: (c[1], c[0]))

        lines = []
        current_line = []
        y_threshold = 10 

        for i, char in enumerate(sorted_chars):
            if not current_line:
                current_line.append(char)
            else:
                # Comparing the y-position with first character 
                # in the current line.
                if abs(char[1] - current_line[0][1]) < y_threshold:
                    current_line.append(char)
                else:
                    # Sort characters in line by x-position
                    current_line.sort(key=lambda c: c[0])
                    lines.append(current_line)
                    current_line = [char]
        
        # The final line.
        if current_line:
            current_line.sort(key=lambda c: c[0])
            lines.append(current_line)
            
        return lines
                
    def detect_characters(self, image):
        """ 
        Character detection using contour finding and line awareness.
        """
        # Thresholding the image and finding contours.
        _, thresh = cv2.threshold(image, 0, 255, 
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Sorting contours from left to right.
        #contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        character_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtering out noise (very small contours).
            if w > 5 and h > 5:
                # Extracting character ROI.
                roi = image[y:y+h, x:x+w]
                character_boxes.append((x, y, w, h, roi))
                
        return character_boxes
    
    def recognize_character(self, char_image):
        """Recognizing a single character."""
        char_pil = Image.fromarray(char_image).convert('L')
        char_tensor = self.transform(char_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(char_tensor)
            _, pred = torch.max(outputs, 1)
            return self.idx_to_class[pred.item()]

    def reconstruct_text(self, lines):
        """
        Recognizing each character and resontructing them into words
        We first turn them into small letters using lower(), so as to
        improve reading accuracy. For example, χ and Χ look quite similar
        and there is a probability that the model predicts the wrong variant
        of the letter.
        Letters are separated with space, and a hyphen case is also being 
        taken into consideration. For example, e-xample is being read as
        example.
        """
        full_text = []
        previous_word_hyphenated = False
        previous_word = []
        
        for line in lines:
            current_line_words = []
            current_word = []
            
            for x, y, w, h, roi in line:
                # Converting into small characters.
                char = self.recognize_character(roi).lower()
                
                if char == self.SPACE_CLASS:
                    # Only adding a word if the list is not empty.
                    if current_word:
                        current_line_words.append(''.join(current_word))
                        current_word = []
                elif char == self.HYPHEN_CLASS:
                    # Only marked as hyphenated if it's part of a word.
                    if current_word:  
                        previous_word_hyphenated = True
                else:
                    current_word.append(char)
            
            # Adding the last word in the line if it exists.
            if current_word:
                current_line_words.append(''.join(current_word))
            
            # Handling hyphenated words from the previous line.
            if previous_word_hyphenated and current_line_words:
                if previous_word:
                    # Combining with the first word of the current line.
                    reconstructed_word = previous_word[-1] + current_line_words[0]
                    previous_word[-1] = reconstructed_word
                    current_line_words = current_line_words[1:]
            
            # Adding words from this line to the full text.
            # Also adding space if not hyphentated.
            if previous_word:
                full_text.extend(previous_word)
                if not previous_word_hyphenated:  
                    full_text.append(' ')
            
            previous_word = current_line_words
            previous_word_hyphenated = False
        
        # Adding any remaining words from the last line.
        if previous_word:
            full_text.extend(previous_word)
        
        return ''.join(full_text).strip()
    
    def recognize_text(self, image_path):
        """Full pipeline for text recognition with line and word handling."""
        image = self.preprocess_image(image_path)
        characters = self.detect_characters(image)
        lines = self.group_characters_into_lines(characters)
        
        return self.reconstruct_text(lines)
        
# ----------------------------------------------------------------------------#
