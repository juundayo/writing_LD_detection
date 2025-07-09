# ----------------------------------------------------------------------------#

import cv2
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------#

class Rectangle:
    def __init__(self, x, y, x2, y2, area):
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2

        self.area = area

# ----------------------------------------------------------------------------#

def rectangleCleaning(rectlist, recaverage):
    '''
    Removing the recntangles that are too
    small besed on an adaptive average score.
    '''
    for rect in rectlist[:]:  # Notice the [:] which creates a copy
        if rect.area < recaverage * 0.2:
            rectlist.remove(rect)
    
# ----------------------------------------------------------------------------#

def preProcessing(myImage):
    grayImg = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray Image', grayImg)
    # cv2.waitKey()
    ret, thresh1 = cv2.threshold(grayImg, 0, 255, 
                                 cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow('After threshold', thresh1)
    #cv2.waitKey()
    print(f'The threshold value applied to the image is: {ret} ')
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, horizontal_kernel, iterations=1)
    horizontal_contours, hierarchy = cv2.findContours(dilation, 
                                                      cv2.RETR_EXTERNAL, 
                                                      cv2.CHAIN_APPROX_NONE)
    im2 = myImage.copy()
    for cnt in horizontal_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 0), 0)

    im2 = character_seg(rect)

    return im2

# ----------------------------------------------------------------------------#

def character_seg(img):
    # Converting the input image into gray scale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding the image.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Applying morphological erosion to remove small artifacts.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    
    # Applying morphological dilation to expand the characters.
    dilated = cv2.dilate(eroded, kernel, iterations=3)
    
    # Finding contours in the image.
    contours, hierarchy = cv2.findContours(dilated,
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterating through each contour and extracting the bounding box.
    iteration = 0
    runningAverage = 0
    reclist = []
    for contour in contours:
        iteration += 1

        (x, y, w, h) = cv2.boundingRect(contour)
        area = (w * h)
        print(w, h, area)

        rec = Rectangle(x, y, x + w, y + h, area)
        reclist.append(rec)

        # Calculating the new average.
        runningAverage = runningAverage + (area - runningAverage) / iteration

    rectangleCleaning(reclist, runningAverage)
    print(f"The average area of the rectangles is: {runningAverage}")

    for rect in reclist:
        cv2.rectangle(img, (rect.x, rect.y), (rect.x2, rect.y2), (0, 255, 0), 2)

    return  img

# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    png = "/home/ml3/Desktop/Thesis/two_mimir.jpg"
    
    image = cv2.imread(png)
    returnImage = preProcessing(image)

    returnImage_rgb = cv2.cvtColor(returnImage, cv2.COLOR_BGR2RGB)

    plt.imshow(returnImage_rgb)
    plt.axis("off")
    plt.savefig("outputPUTPUTPUT.png", bbox_inches='tight', pad_inches=0)
