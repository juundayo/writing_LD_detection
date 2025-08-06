# ----------------------------------------------------------------------------#

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------------------------------------------------------#

INPUT_FOLDER = "/home/ml3/Desktop/Thesis/TextBlocks"
OUTPUT_FOLDER = "/home/ml3/Desktop/Thesis/LetterCrops"
LINE_DATA_FILE = "/home/ml3/Desktop/Thesis/line_data.json"

# ----------------------------------------------------------------------------#

class Rectangle:
    def __init__(self, x, y, x2, y2, area):
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.area = area
        self.center_x = (x + x2) / 2
        self.center_y = (y + y2) / 2

# ----------------------------------------------------------------------------#

class Word:
    def __init__(self):
        self.characters = []
        self.x_min = float('inf')
        self.y_min = float('inf')
        self.x_max = 0
        self.y_max = 0
    
    def add_character(self, rect):
        self.characters.append(rect)
        self.x_min = min(self.x_min, rect.x)
        self.y_min = min(self.y_min, rect.y)
        self.x_max = max(self.x_max, rect.x2)
        self.y_max = max(self.y_max, rect.y2)
    
    def get_bbox(self):
        return (self.x_min, self.y_min, self.x_max, self.y_max)

# ----------------------------------------------------------------------------#

def pre_processing(myImage, im_average=None):
    grayImg = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(grayImg, 0, 255, 
                                cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    print(f'The threshold value applied to the image is: {ret} ')

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, horizontal_kernel, iterations=1)
    horizontal_contours, _ = cv2.findContours(dilation, 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_NONE)
    im2 = myImage.copy()
    
    return character_segmentation(im2, im_average)

# ----------------------------------------------------------------------------#

def cluster_characters_to_words(rectangles):
    """
    Clusters individual characters into words based on dynamic proximity threshold.
    """
    if not rectangles:
        return []
    
    # Sort rectangles left to right
    sorted_rects = sorted(rectangles, key=lambda r: r.x)
    
    # Calculate average character width
    avg_width = np.mean([r.x2 - r.x for r in sorted_rects])
    
    # Dynamic gap threshold (e.g., 0.5 * average width)
    dynamic_gap_threshold = avg_width * 0.5
    
    words = []
    current_word = Word()
    
    for rect in sorted_rects:
        if not current_word.characters:
            current_word.add_character(rect)
        else:
            last_char = current_word.characters[-1]
            gap = rect.x - last_char.x2
            
            # Use dynamic threshold with minimum fallback
            if gap < max(dynamic_gap_threshold, 10):  # Never less than 10px
                current_word.add_character(rect)
            else:
                words.append(current_word)
                current_word = Word()
                current_word.add_character(rect)
    
    if current_word.characters:
        words.append(current_word)
    
    return words

# ----------------------------------------------------------------------------#

def character_segmentation(img, im_average=None):
    # Grayscale conversion and thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []    
    running_average = 0
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        rect = Rectangle(x, y, x + w, y + h, area)
        rectangles.append(rect)
        running_average = running_average + (area - running_average) / (i + 1)

    # Initial area-based filtering
    filter_threshold = im_average * 0.33 if im_average is not None else running_average * 0.33
    filtered_rectangles = [rect for rect in rectangles if rect.area >= filter_threshold]

    # Baseline-based tonos removal.
    if im_average is not None:
        # Calculating the average baseline (bottom y-coordinate) of characters.
        avg_baseline = np.mean([rect.y2 for rect in filtered_rectangles])
        
        # Calculating the average character height.
        avg_height = np.mean([rect.y2 - rect.y for rect in filtered_rectangles])
        
        # Defining threshold for "too high".
        height_threshold = avg_baseline - (0.5 * avg_height)
        
        # Filtering out characters significantly above the baseline.
        filtered_rectangles = [
            rect for rect in filtered_rectangles
            if rect.y2 > height_threshold  
        ]

    # Clustering characters into words
    words = cluster_characters_to_words(filtered_rectangles)
    
    # Visualization
    vis_img = img.copy()
    for rect in filtered_rectangles:
        cv2.rectangle(vis_img, (rect.x, rect.y), (rect.x2, rect.y2), (0, 255, 0), 2)
    
    # Prepare word data
    word_data = []
    for i, word in enumerate(words):
        x1, y1, x2, y2 = word.get_bbox()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        word_data.append({
            'word_id': i,
            'bbox': (x1, y1, x2, y2),
            'characters': [(c.x, c.y, c.x2, c.y2) for c in word.characters]
        })
    
    return vis_img, word_data, running_average

# ----------------------------------------------------------------------------#

def process_image_block(image_block, im_average):
    """Main processing function to be called externally."""
    segmented_img, word_data, _ = pre_processing(image_block, im_average)

    return segmented_img, word_data

# ----------------------------------------------------------------------------#

def get_image_average(full_image):
    _, _, running_average = pre_processing(full_image)

    return running_average

# ----------------------------------------------------------------------------#

def testing():
    png = "/home/ml3/Desktop/Thesis/BlockImages/block_1.png"
    png2 = "/home/ml3/Desktop/Thesis/two_mimir.jpg"

    image = cv2.imread(png)
    image2 = cv2.imread(png2)

    average = get_image_average(image2)
    segmented_img, word_data = process_image_block(image_block=image, im_average=average)
    segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
    
    # Saving and showing results.
    plt.imshow(segmented_img_rgb)
    plt.axis("off")
    plt.savefig("0_segmented.png", bbox_inches='tight', pad_inches=0)
    plt.show()

# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    testing()
