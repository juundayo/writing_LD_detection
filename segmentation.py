# ----------------------------------------------------------------------------#

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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

def rectangle_cleaning(rectlist, recaverage):
    return [rect for rect in rectlist if rect.area >= recaverage * 0.2]

def pre_processing(myImage):
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
    
    return character_segmentation(im2)

def cluster_characters_to_words(rectangles, max_gap=30):
    """
    Clusters individual characters
    into words based on proximity.
    """
    if not rectangles:
        return []
    
    # Sorting rectangles left to right.
    sorted_rects = sorted(rectangles, key=lambda r: r.x)
    
    words = []
    current_word = Word()
    
    for i, rect in enumerate(sorted_rects):
        if not current_word.characters:
            current_word.add_character(rect)
        else:
            # Calculating the gap between current 
            # character and previous character.
            last_char = current_word.characters[-1]
            gap = rect.x - last_char.x2
            
            if gap < max_gap:
                current_word.add_character(rect)
            else:
                words.append(current_word)
                current_word = Word()
                current_word.add_character(rect)
    
    if current_word.characters:
        words.append(current_word)
    
    return words

# ----------------------------------------------------------------------------#

def character_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))

    # Removing noise.
    eroded = cv2.erode(thresh, kernel, iterations=1)

    # Expanding characters.
    dilated = cv2.dilate(eroded, kernel, iterations=3)
    
    contours, _ = cv2.findContours(dilated, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    iteration = 0
    runningAverage = 0
    rectangles = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        rect = Rectangle(x, y, x + w, y + h, area)
        rectangles.append(rect)
        
        runningAverage = runningAverage + (area - runningAverage) / (iteration + 1)
        iteration += 1
    
    # Filtering small rectangles (noise, tonos and dialytics).
    rectangles = rectangle_cleaning(rectangles, runningAverage)
    
    # Clustering characters into words.
    words = cluster_characters_to_words(rectangles)
    
    vis_img = img.copy()
    
    # Drawing character boxes (green).
    for rect in rectangles:
        cv2.rectangle(vis_img, (rect.x, rect.y), (rect.x2, rect.y2), (0, 255, 0), 2)
    
    # Drawing word boxes (blue).
    word_data = []
    for i, word in enumerate(words):
        x1, y1, x2, y2 = word.get_bbox()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Storing word data with character coordinates.
        word_data.append({
            'word_id': i,
            'bbox': (x1, y1, x2, y2),
            'characters': [(c.x, c.y, c.x2, c.y2) for c in word.characters]
        })
    
    return vis_img, word_data

# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    png = "/home/ml3/Desktop/Thesis/Screenshot_7.png"
    image = cv2.imread(png)
    
    segmented_img, word_data = pre_processing(image)
    segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
    
    # Saving and showing results.
    plt.imshow(segmented_img_rgb)
    plt.axis("off")
    plt.savefig("thesis_crop_test.png", bbox_inches='tight', pad_inches=0)
    plt.show()
