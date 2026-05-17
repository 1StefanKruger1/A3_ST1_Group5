# import numpy as np
import cv2 as cv

from src.config import IMAGE_SIZE
class ImagePreprocessor:
    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size
    def transform(self,file_path):
            #reads an image
            image = cv.imread(str(file_path), cv.IMREAD_GRAYSCALE)
            #checking if there is no image or it cant read
            if image is None:
                raise FileNotFoundError("File not found or Could not be read, Pls choose different options. ")


            resized = cv.resize(image, self.image_size)
            normalized = resized.astype("float")/255.0
            return normalized.flatten()




