import cv2

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as vF
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt



def main():
    # Open the first webcam device found by your system
    cam = cv2.VideoCapture(0)
    
    ch = 10
    # r = torch.randn(1, 3, 5, 5)
    # r = torch.normal(mean=torch.zeros(1, 3, 5, 5), std=2)
    r = torch.tensor([[[[ 0.7917, -2.4136,  0.7076,  3.0385, -1.4910],
          [-1.8450,  2.1870, -1.4583, -1.8226, -1.2093],
          [ 1.8683,  0.6509, -2.6384,  0.1825,  2.0432],
          [ 0.0478, -0.2022,  1.2576, -1.5327,  1.5264],
          [ 0.1937, -1.9253,  0.3620, -1.2662,  2.0145]],

         [[ 1.2917,  0.0658, -0.9486, -0.9843,  1.1612],
          [ 0.2130, -1.0223, -2.5381,  2.6797,  0.2830],
          [-0.7533, -0.5361, -0.1855, -0.8605,  0.6685],
          [ 1.2721, -1.9571,  0.8976, -2.7793,  1.5025],
          [-0.2754,  0.0170,  1.9445, -1.1021,  2.3567]],

         [[-1.0516,  0.6095,  0.1762, -1.8851, -3.3705],
          [-0.3485, -3.8046, -0.5211, -0.0718, -1.5302],
          [ 3.3458,  1.9384,  1.2905, -3.4550,  1.0836],
          [-0.7728, -1.5056, -1.8409, -1.1651, -0.2800],
          [ 1.0632,  0.2673,  1.0127,  2.3651,  1.9630]]]])

    sx = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sy = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # r = torch.fill(torch.zeros(1, 3, 2, 2), 0.2)
    # print(r)
    # Run continuous loop while webcam device is being used
    while(cam.isOpened()):

        # grab the next video frame
        # cv2.VideoCapture.read returns 2 values
        # retval: returns true or false if the read
        # operation was successful or not, we discard
        # it here by using the _ variable name
        # image: this returns a matrix containing the
        # pixel values of the image, in we assign this
        # value to frame
        _, frame = cam.read()

        # cv2.cvtColor(compfr, cv2.COLOR_RGB2GRAY)

        ft = ToTensor()(frame)
      
        # ft = ft.squeeze(dim=2)
        # ft = F.linear(ft, torch.ones(1440, 640))

        # ft = F.max_pool2d(ft, kernel_size=(2,2)) 

        # ft = conv1(ft) 
        # ft = conv2(ft)
        # ft = vF.invert(ft)
        # ft = vF.adjust_brightness(ft, 0.5)
        # ft = vF.adjust_contrast(ft, 3.0)
        # ft = F.conv2d(ft, r)
        # ft = F.conv2d(ft, torch.randn(1, 3, 5, 5))
        # ft = nn.Conv2d(3, 2, kernel_size=1)(ft)
        # ft = F.fractional_max_pool2d(ft, kernel_size=(10,10), output_ratio=(0.9,0.9))
        # ft = F.conv_transpose2d(ft, r)
        # ft = ft.round(decimals=1)
        ft = vF.rgb_to_grayscale(ft, 3)
        ft = F.conv2d(ft, sx)
        ft = F.conv2d(ft, xy)
        # ft = F.relu(ft)
        # print(ft)

        ft = ToPILImage()(ft)

        frame = np.array(ft)

        frame = cv2.resize(frame, (1280, 960), interpolation=0)
        
        # create or write to a window named "Hello OpenCV" 
        # and display our captures frame
        cv2.imshow('Hello OpenCV', frame)

        # wait for the user to press "q" before exiting
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.imwrite("exp.png", frame)
            break

    # free the webcam devicce
    cam.release()
    # close "Hello OpenCV"
    cv2.destroyWindow('Brain')

if __name__ == "__main__":
    main()
