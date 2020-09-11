import torchvision
import tensorflow as tf
import os
import torch
import numpy as np
import matplotlib.image as mpimg
import cv2


def get_torch_model():  
    os.environ['TORCH_MODEL_ZOO'] = '../../models'
    resnet_18 = torchvision.models.resnet18(pretrained=True)
    return resnet_18

if __name__ == "__main__":
    model = get_torch_model()
    model.eval()

    image1 = mpimg.imread('../images/cat_1.jpeg')
    image1 = cv2.resize(image1, (224, 224))
    images = np.array([image1]) / 255.0

    data = np.transpose(images, (0, 3, 1, 2))
    data = torch.from_numpy(data).float()

    # data = torch.randn(1, 3, 224, 224) # Load your data here, this is just dummy data
    output = model(data)
    # prediction = torch.argmax(output)
    prediction = output.data.numpy()[0]
    print(np.argmax(prediction))


