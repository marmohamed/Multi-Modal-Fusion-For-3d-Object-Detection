
import numpy as np
import cv2

class ImageReader:

    def __init__(self, image_path, translate_x = 0, translate_y = 0, image_size=(370, 1224)):
        self.image_path = image_path
        self.image_size = image_size
        self.translate_x = translate_x
        self.translate_y = translate_y

    
    def read_image(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation = cv2.INTER_AREA)
        if self.translate_x > 0:
            image [:, abs(self.translate_x):self.image_size[1]]= image[:, :self.image_size[1]-abs(self.translate_x)]
            image[:, :abs(self.translate_x)] = 0
        elif self.translate_x < 0:
            image [:, :self.image_size[1]-abs(self.translate_x)]= image[:, abs(self.translate_x):self.image_size[1]]
            image[:, self.image_size[1]-abs(self.translate_x):] = 0

        if self.translate_y > 0:
            image [abs(self.translate_y):self.image_size[0], :]= image[:self.image_size[0]-abs(self.translate_y), :]
            image[:abs(self.translate_y), :] = 0
        elif self.translate_y < 0:
            image [:self.image_size[0]-abs(self.translate_y), :]= image[abs(self.translate_y):self.image_size[0], :]
            image[self.image_size[0]-abs(self.translate_y):, :] = 0
        image = image / 255.
        return image