import cv2

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as vF
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import sys



def main():
    # Open the first webcam device found by your system
    cam = cv2.VideoCapture(0)
    
    ch = 10
    # r = torch.randn(1, 3, 5, 5)
    # r = torch.normal(mean=torch.zeros(1, 3, 5, 5), std=2)


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

        ft = F.max_pool2d(ft, kernel_size=(int(sys.argv[1]),int(sys.argv[1]))) 

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
        # ft = vF.rgb_to_grayscale(ft, 3)
        # ft = F.conv2d(ft, r)
        ft = ft.round(decimals=int(sys.argv[2]))
        # ft = F.conv2d(ft, sx)
        # ft = F.conv2d(ft, xy)
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
