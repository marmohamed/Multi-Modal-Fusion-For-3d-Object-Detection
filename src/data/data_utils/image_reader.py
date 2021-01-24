
import numpy as np
import cv2
import imutils


class ImageReader:

    def __init__(self, image_path, translate_x = 0, translate_y = 0, translate_z=0, ang=0, fliplr=False, image_size=(370, 1224)):
        self.image_path = image_path
        self.image_size = image_size
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.translate_z = translate_z
        self.ang = ang
        self.fliplr = fliplr

    
    def read_image(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation = cv2.INTER_AREA)

        self.translate_y = int((self.translate_y / 80.) * self.image_size[1])
        self.translate_z = int((self.translate_z / 4.) * self.image_size[0])

        self.translate_x = self.translate_x / 70.
        if self.translate_x != 0:
            self.translate_x = 1 + self.translate_x

        if self.translate_y > 0:
            image [:, abs(self.translate_y):self.image_size[1]]= image[:, :self.image_size[1]-abs(self.translate_y)]
            image[:, :abs(self.translate_y)] = 0
        elif self.translate_y < 0:
            image [:, :self.image_size[1]-abs(self.translate_y)]= image[:, abs(self.translate_y):self.image_size[1]]
            image[:, self.image_size[1]-abs(self.translate_y):] = 0

        if self.translate_z > 0:
            image [abs(self.translate_z):self.image_size[0], :]= image[:self.image_size[0]-abs(self.translate_z), :]
            image[:abs(self.translate_z), :] = 0
        elif self.translate_z < 0:
            image [:self.image_size[0]-abs(self.translate_z), :]= image[abs(self.translate_z):self.image_size[0], :]
            image[self.image_size[0]-abs(self.translate_z):, :] = 0

        if self.translate_x != 0:
           image = self.Zoom(image, self.translate_x)

        if self.fliplr:
            image = np.fliplr(image)

        if self.ang > 0:
            image = self.rotate(image, self.ang)

        image = image / 255.
        return image

    
    def Zoom(self, img, zoom_scale):
        rows, cols, dim = img.shape
        cv2Object = cv2.resize(img, (int(cols * zoom_scale), int(rows * zoom_scale)), interpolation = cv2.INTER_AREA)
        center = (rows//2, cols//2)
       
        cv2Object = cv2Object[max(0, center[0]-rows//2):min(cv2Object.shape[0], center[0] + (rows-rows//2)),\
                              max(0, center[1] - cols//2):min(cv2Object.shape[1], center[1] + (cols-cols//2)), :]

        new_size = cv2Object.shape

        delta_w = cols - new_size[1]
        delta_h = rows - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(cv2Object, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=[0, 0, 0])

        return new_im


    # def Zoom(self, cv2Object, zoomSize):
    #     # https://github.com/CJoseFlores/python-OpenCV-Zoom/blob/master/ZoomTest.py
    #     # Resizes the image/video frame to the specified amount of "zoomSize".
    #     # A zoomSize of "2", for example, will double the canvas size
    #     cv2Object = imutils.resize(cv2Object, width=(int(zoomSize * cv2Object.shape[1])))
    #     # center is simply half of the height & width (y/2,x/2)
    #     center = (cv2Object.shape[0]/2,cv2Object.shape[1]/2)
    #     # cropScale represents the top left corner of the cropped frame (y/x)
    #     cropScale = (int(center[0]/zoomSize), int(center[1]/zoomSize))
    #     # The image/video frame is cropped to the center with a size of the original picture
    #     # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
    #     # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
    #     cv2Object = cv2Object[cropScale[0]:(center[0] + cropScale[0]), cropScale[1]:(center[1] + cropScale[1]), :]
    #     return cv2Object


    def rotate(self, img, ang):
        rows, cols, dim = img.shape
        angle = np.radians(ang)
        M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])
        rotated_img = cv2.warpPerspective(img, M, (int(cols),int(rows)))
        return rotated_img