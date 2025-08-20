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

def remove_horizontal_lines(thresh):
    """
    Detects and removes horizontal notebook lines using morphological operations.
    Returns a cleaned threshold image.
    """
    # Define a long horizontal kernel based on image width
    cols = thresh.shape[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // 20, 1))
    
    # Detect horizontal lines
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Subtract lines from the thresholded image
    cleaned = cv2.subtract(thresh, detected_lines)
    
    return cleaned

# ----------------------------------------------------------------------------#

def pre_processing(myImage):
    grayImg = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(grayImg, 0, 255, 
                                cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    #print(f'The threshold value applied to the image is: {ret} ')

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, horizontal_kernel, iterations=1)
    horizontal_contours, _ = cv2.findContours(dilation, 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_NONE)
    im2 = myImage.copy()
    
    return character_segmentation(im2)

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
    dynamic_gap_threshold = avg_width * 0.77
    
    words = []
    current_word = Word()
    
    for rect in sorted_rects:
        if not current_word.characters:
            current_word.add_character(rect)
        else:
            last_char = current_word.characters[-1]
            gap = rect.x - last_char.x2
            
            # Uses dynamic threshold with minimum fallback.
            if gap < max(dynamic_gap_threshold, 10):  # Never less than 10px.
                current_word.add_character(rect)
            else:
                words.append(current_word)
                current_word = Word()
                current_word.add_character(rect)
    
    if current_word.characters:
        words.append(current_word)
    
    return words

# ----------------------------------------------------------------------------#

def split_large_boxes(rectangles, thresh_img):
    """
    Splits oversized rectangles at their thinnest vertical 
    connection point. Returns an updated list of rectangles.
    """
    # Calculates median dimensions for thresholds.
    areas = [rect.area for rect in rectangles]
    widths = [rect.x2 - rect.x for rect in rectangles]
    heights = [rect.y2 - rect.y for rect in rectangles]

    median_area = np.median(areas)
    median_width = np.median(widths)
    meadian_height = np.median(heights)

    # Splitting thresholds.
    area_threshold = median_area * 1.5
    width_threshold = median_width * 1.7

    new_rectangles = []
    for rect in rectangles:
        w = rect.x2 - rect.x
        h = rect.y2 - rect.y

        # Checking if the box needs splitting.
        if rect.area > area_threshold and w > width_threshold:
            # Extracting the region from the thresholded image.
            crop = thresh_img[rect.y:rect.y2, rect.x:rect.x2]
            if crop.size == 0:
                new_rectangles.append(rect)
                continue
            
            projection = np.sum(crop, axis=0) // 255
            width = crop.shape[1]

            left_bound = max(1, int(width * 0.15))
            right_bound = min(width - 1, int(width * 0.95))
            valid_region = projection[left_bound:right_bound]

            if len(valid_region) == 0:
                new_rectangles.append(rect)
                continue

            # Finding the minimal connection point.
            min_idx = np.argmin(valid_region)
            min_val = valid_region[min_idx]
            abs_idx = left_bound + min_idx

            max_proj = np.max(projection)
            aspect_ratio = w / float(h)

            if (aspect_ratio > 1.4 and rect.area > 1.5 * median_area
                and min_val < 0.3 * max_proj):
                # Left rectangle.
                left_rect = Rectangle(rect.x, rect.y, 
                                      rect.x + abs_idx, rect.y2, abs_idx * h)
                # Right rectange.
                right_rect = Rectangle(rect.x + abs_idx, rect.y, 
                                       rect.x2, rect.y2, (w - abs_idx) * h)
                
                new_rectangles.extend([left_rect, right_rect])
            else:
                new_rectangles.append(rect)
        else:
            new_rectangles.append(rect)

    return new_rectangles

# ----------------------------------------------------------------------------#

def remove_tonos_by_vertical_relation(rectangles):
    """
    Removes tonos by checking if there is a smaller box directly above a letter box.
    """
    if not rectangles:
        return rectangles

    # Sort rectangles bottom-to-top for stable checking
    rectangles = sorted(rectangles, key=lambda r: r.y)

    to_remove = set()
    median_height = np.median([r.y2 - r.y for r in rectangles])

    for base in rectangles:
        base_height = base.y2 - base.y
        base_center_x = (base.x + base.x2) / 2

        # Look for smaller boxes above this base
        for candidate in rectangles:
            if candidate is base:
                continue

            # candidate must be above base but not too far
            if candidate.y2 <= base.y and (base.y - candidate.y2) <= median_height:
                # Horizontal overlap (allow slight offset)
                horizontal_overlap = not (candidate.x2 < base.x - 2 or candidate.x > base.x2 + 2)

                # Must be significantly smaller (likely tonos)
                if (candidate.y2 - candidate.y) < 0.6 * base_height and horizontal_overlap:
                    to_remove.add(candidate)
    
    # Return rectangles except the ones marked as tonos
    return [r for r in rectangles if r not in to_remove]

# ----------------------------------------------------------------------------#

def character_segmentation(img):
    # Grayscale conversion and thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    thresh = remove_horizontal_lines(thresh)

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
    filter_threshold = running_average * 0.20
    filtered_rectangles = [rect for rect in rectangles if rect.area >= filter_threshold]

    # Tonos removal.
    filtered_rectangles = remove_tonos_by_vertical_relation(filtered_rectangles)
    
    # Splitting large boxes after all filtering.
    filtered_rectangles = split_large_boxes(filtered_rectangles, thresh)

    # Clustering characters into words
    words = cluster_characters_to_words(filtered_rectangles)
    
    # Visualization
    vis_img = img.copy()
    for rect in filtered_rectangles:
        cv2.rectangle(vis_img, (rect.x, rect.y), (rect.x2, rect.y2), (0, 255, 0), 2)
    
    # Preparing word data.
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

def process_image_block(image_block):
    """Main processing function to be called externally."""
    segmented_img, word_data, _ = pre_processing(image_block)

    return segmented_img, word_data

# ----------------------------------------------------------------------------#

def testing():
    #png = "/home/ml3/Desktop/Thesis/BlockImages/block_3.png"
    #png = "/home/ml3/Desktop/Thesis/Screenshot_15.png"
    png = "/home/ml3/Desktop/Thesis/BlockImages/block_3.png"

    image = cv2.imread(png)

    segmented_img, word_data = process_image_block(image_block=image)

    segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
    
    # Saving and showing results.
    plt.imshow(segmented_img_rgb)
    plt.axis("off")
    plt.savefig("0_segmented.png", bbox_inches='tight', pad_inches=0)
    plt.show()

# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    testing()
