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
    
    def preprocess_image(self, image_path):
        """Loads and preprocesses the image for character detection."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image
    
    def detect_characters(self, image):
        """Simple character detection using contour finding."""
        # Thresholding the image.
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Finding contours.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sorting contours from left to right.
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
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
    
    def recognize_text(self, image_path):
        """Full pipeline for text recognition!"""
        # Preprocesses the image.
        image = self.preprocess_image(image_path)
        
        # Detecting characters.
        characters = self.detect_characters(image)
        
        # Recognizing each character
        recognized_text = []
        for x, y, w, h, roi in characters:
            char = self.recognize_character(roi)
            recognized_text.append(char)
            
        # Combining characters into words/sentences.
        # For now, we'll test joining with space.
        return ' '.join(recognized_text)
    
# ----------------------------------------------------------------------------#
