# ----------------------------------------------------------------------------#

import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import numpy as np

from ocr_model import LetterOCR
from data_loading import class_mapping

# ----------------------------------------------------------------------------#

SHOW_PREDICTIONS = True

# ----------------------------------------------------------------------------#

class GreekTextRecognizer:
    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_model(model_path, len(class_mapping))
        
        # Inverse mapping from index to class name.
        self.classes = sorted(class_mapping.values())
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        self.greek_dictionary = self.load_greek_dictionary(
            '/home/ml3/Desktop/Thesis/.venv/Data/filtered_dictionary.dic'
        ) 
        
        self.transform = transforms.Compose([
            transforms.Resize((45, 80)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.HYPHEN_CLASS = 'hyphen'
    
    def load_model(self, model_path, num_classes):
        model = LetterOCR(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)

        return model
    
    def visualize_detections(self, image, characters, show_predictions=False):
        """Displaying the input image with detected character bounding boxes."""
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h, roi) in characters:
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if show_predictions:
                char, conf = self.recognize_character(roi)

                label = f"{char} ({conf:.2f})".encode('utf-8').decode('utf-8')

                image = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                draw.text((x, y - 5), label, fill=(255, 0, 0))
        
        cv2.imshow("Character Detections", display_img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        return key 
    
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
    
    def preprocess_image(self, image_path):
        """
        Loads and preprocesses the image for character detection.
        Using adaptive threshold for better performance.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Adaptice histogram equalization for contrast.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        image = clahe.apply(image)

        image = cv2.medianBlur(image, 3)
        
        # Adaptive thresholding.
        image = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10
        )
        
        # Morphological operations to clean up!
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        return image
    
    def preprocess_character(self, char_image):
        # Creating a padded image.
        h, w = char_image.shape
        size = max(h, w) + 10
        padded = np.zeros((size, size), dtype=np.uint8)

        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = char_image

        blurred = cv2.GaussianBlur(padded, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        resized = cv2.resize(thresh, (80, 45),
                              interpolation=cv2.INTER_AREA)
        
        return resized

    def line_grouping(self, characters):
        '''
        Groups characters into lines based on their y-coordinate.
        We use the median character height to calculate the height
        of each line dynamically.
        '''

        if not characters:
            return []
        
        heights = [h for _, _, _, h, _ in characters]
        median_height = np.median(heights) if heights else 20
        
        # Sorting characters by y and x.
        sorted_chars = sorted(characters, key=lambda c: (c[1], c[0]))
        
        lines = []
        current_line = []
        
        for char in sorted_chars:
            if not current_line:
                current_line.append(char)
            else:
                # Comparing with the first character in the current line.
                ref_char = current_line[0]
                y_diff = abs(char[1] - ref_char[1])
                
                # Dynamic threshold based on character height.
                if y_diff < median_height * 0.6: 
                    current_line.append(char)
                else:
                    # Sorting the current line by x before adding.
                    current_line.sort(key=lambda c: c[0])
                    lines.append(current_line)
                    current_line = [char]
        
        if current_line:
            current_line.sort(key=lambda c: c[0])
            lines.append(current_line)
        
        # Sorting the lines by their average y position.
        lines.sort(key=lambda line: np.mean([c[1] for c in line]))
        
        return lines
                
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
        word_stats = defaultdict(int)
        confidence_scores = []
        
        for line in lines:
            current_line_words = []
            current_word = []
            current_confidences = []
            
            for x, y, w, h, roi in line:
                # Converting into small characters.
                char, conf = self.recognize_character(roi)
                char = char.lower()
                confidence_scores.append(conf)
                
                if char == ' ':
                    # Only adding a word if the list is not empty.
                    if current_word:
                        word = ''.join(current_word)
                        avg_conf = sum(current_confidences)/len(current_confidences)

                        marked_word = self.mark_unknown(word, avg_conf)
                        current_line_words.append(marked_word)
                        current_word = []
                        current_confidences = []
                elif char == self.HYPHEN_CLASS:
                    # Only marked as hyphenated if it's part of a word.
                    if current_word:  
                        previous_word_hyphenated = True
                else:
                    current_word.append(char)
                    current_confidences.append(conf)
            
            # Adding the last word in the line if it exists.
            if current_word:
                word = ''.join(current_word)
                avg_conf = sum(current_confidences)/len(current_confidences) if current_confidences else 0
                current_line_words.append(self.mark_unknown(word, avg_conf))
            
            # Handling hyphenated words from the previous line.
            if previous_word_hyphenated and current_line_words:
                if previous_word:
                    # Combining with the first word of the current line.
                    joined_word = previous_word[-1] + current_line_words[0]

                    # Checking if the joined word exists in the dictionary.
                    if joined_word.lower() in self.greek_dictionary:
                        previous_word[-1] = joined_word
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
        
        avg_confidence = sum(confidence_scores)/len(confidence_scores) if confidence_scores else 0

        return {
            'text': ''.join(full_text).strip(),
            "confidence": avg_confidence
        }
    
    def mark_unknown(self, word, confidence=0):
        if (word.lower() not in self.greek_dictionary) or (confidence < 0.7):
            return f"*{word}*"
        return word

    def recognize_text(self, image_path):
        """Full pipeline for text recognition with line and word handling."""
        image = self.preprocess_image(image_path)
        characters = self.detect_characters(image)

        self.visualize_detections(image, characters, 
                                  show_predictions=SHOW_PREDICTIONS)
    
        # Visualizations for each character.
        for i, (x, y, w, h, roi) in enumerate(characters):
            cv2.imshow(f"Character {i}", roi)
            cv2.waitKey(0)
            
            # Recognizing the character.
            char, conf = self.recognize_character(roi)
            print(f"Recognized: {char} (confidence: {conf:.2f})")
            
            # Showing the recognition result on original image.
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_img, char, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Recognition Progress", display_img)
            cv2.waitKey(0)
        
            cv2.destroyAllWindows()

        lines = self.line_grouping(characters)
        result = self.reconstruct_text(lines)

        # Calculating accuracy stats!
        recognized_text = result['text']
        words = recognized_text.split()

        known_words = [w for w in words if not w.startswith('*')]
        accuracy = len(known_words)/len(words) if words else 0
        
        return {
            'text': recognized_text,
            'accuracy': accuracy,
            'char_count': len(''.join(words)),
            'word_count': len(words),
            'unknown_words': [w.strip('*') for w in words if w.startswith('*')],
            'confidence': result['confidence']
        }
        

    def detect_characters(self, image):
        """ 
        Character detection using contour finding and line awareness.
        A rectangle is drawn around each character found.
        """
        # Thresholding the image and finding contours.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8
        )
                
        character_boxes = []
        min_height = image.shape[0] * 0.02  
        max_height = image.shape[0] * 0.3 
        aspect_ratio_range = (0.2, 3.0)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if (h >= min_height and h <= max_height and 
                w/h >= aspect_ratio_range[0] and w/h <= aspect_ratio_range[1]):

                component = (labels == i).astype("uint8") * 255
                roi = component[y:y+h, x:x+w]

                if np.mean(roi) > 15:
                    character_boxes.append((x, y, w, h, roi))

        character_boxes = self.merge_character_boxes(character_boxes)

        return character_boxes
    
    def merge_character_boxes(self, boxes, x_threshold=0.2, y_threshold=0.5):
        '''
        Merging boxes that are close horizontally or vertically.
        For exmaple tonos and letters that look like two letters combined.
        '''
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

        merged = []
        current = list(boxes[0])

        for box in boxes[1:]:
            x, y, w, h, roi = box
            curr_x, curr_y, curr_w, curr_h, curr_roi = current
        
            # Overlap and distance metrics.
            x_dist = abs(x - (curr_x + curr_w))
            y_dist = abs(y - (curr_y + curr_h))
            avg_height = (h + curr_h) / 2
            
            # Checking if boxes should be merged.
            if (x_dist < avg_height * x_threshold and 
                y_dist < avg_height * y_threshold):
                
                # Merging the boxes.
                new_x = min(x, curr_x)
                new_y = min(y, curr_y)
                new_w = max(x + w, curr_x + curr_w) - new_x
                new_h = max(y + h, curr_y + curr_h) - new_y
                
                # Creating the new ROI.
                new_roi = np.zeros((new_h, new_w), dtype=np.uint8)
                new_roi[curr_y-new_y:curr_y-new_y+curr_h, 
                    curr_x-new_x:curr_x-new_x+curr_w] = curr_roi
                new_roi[y-new_y:y-new_y+h, x-new_x:x-new_x+w] = roi
                
                current = [new_x, new_y, new_w, new_h, new_roi]
            else:
                merged.append(tuple(current))
                current = list(box)
        
        merged.append(tuple(current))

        return merged

    def recognize_character(self, char_image):
        """
        Recognizing a single character.
        Returns the character itself and the 
        respective confidence of the prediction.
        """
        processed_char = self.preprocess_character(char_image)

        char_pil = Image.fromarray(processed_char).convert('L')
        char_tensor = self.transform(char_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(char_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            char = self.idx_to_class[pred.item()]
            
            return char.lower(), conf.item()
        

# ----------------------------------------------------------------------------#

if __name__ == '__main__':
    recognizer = GreekTextRecognizer("ocr_model.pth", "cuda")
    img_path = "/home/ml3/Desktop/Thesis/.venv/model_test.jpg"
    result = recognizer.recognize_text(img_path)

    print("\nFinal Results:")
    print(f"Recognized Text: {result['text']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Character Count: {result['char_count']}")
    print(f"Word Count: {result['word_count']}")
    print(f"Unknown Words: {result['unknown_words']}")
