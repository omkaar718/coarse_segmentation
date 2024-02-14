from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# use gpu if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


"""
Prerequisites 
1. Define the model class
2. Define the resize with pad transformation class
"""
class MobileNetV3WithConv(nn.Module):
    def __init__(self):
        super(MobileNetV3WithConv, self).__init__()
        
        # Load the MobileNetV3-small model
        self.mobilenet_v3 = models.mobilenet_v3_small(weights = 'DEFAULT')
        self.mobilenet_v3 = self.mobilenet_v3.features[0:9]
        self.conv = nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.mobilenet_v3(x)
        x = self.conv(x)
        x = self.sigmoid(x)

        return x


class CustomResizeWithPad():
    def __init__(self, target_size = (240, 240)):
        self.target_size = target_size

    def __call__(self, image):
        # Calculate padding
        width, height = image.size
        ratio = min(self.target_size[0] / width, self.target_size[1] / height)
        new_size = (round(width * ratio), round(height * ratio))
        padding = (int((self.target_size[0] - new_size[0]) / 2), int((self.target_size[1] - new_size[1]) / 2))

        # Resize image with padding
        resized_image = ImageOps.pad(image.resize(new_size, Image.ANTIALIAS), self.target_size)
        return resized_image


def main():
    # Instantiate the model
    model = MobileNetV3WithConv().to(DEVICE)

    # Load model weights
    model_weights_path = 'segmentation_model_epoch_11.pth'
    model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))

    # Load image
    image_path = 'orig_images/1_Handshaking_Handshaking_1_165.png'
    input_img = Image.open(image_path)

    # image transformation
    image_transform = transforms.Compose([
        CustomResizeWithPad(target_size = (240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3298, 0.3051, 0.2864], std=[0.3051, 0.2883, 0.2805])
    ])

    # transform image and add extra 0th dimension for batch (using unsqueeze()) to make the image dimensions compatible with PyTorch model
    transformed_image = image_transform(input_img).unsqueeze(0)

    # Inference
    prediction = model(transformed_image).squeeze() # This is a 15x15 array (PyTorch tensor) 
    print('\nDone!\n')


if __name__ == '__main__':
    main()