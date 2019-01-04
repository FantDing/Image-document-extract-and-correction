import cv2
import numpy as np
from transform import cvtColor, warpAffine
from corner_detection import detect_corners
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = "./data/1.jpg"
    # path = "./data/000026.jpg"
    # path = './data/000872.jpg'
    # path = './data/001201.jpg'
    # path = './data/001402.jpg'
    # path = './data/001552.jpg'
    src_img = cv2.imread(path)
    gray_img = cvtColor(src_img)
    detect_img, detected_corner = detect_corners(gray_img)

    tar_img = warpAffine(src_img, detected_corner)
    plt.imshow(detect_img)
    plt.show()
    # cv2.imshow('Window', np.uint8(tar_img))
    # cv2.waitKey(0)
    # cv2.imshow('dst', detect_img)
    # cv2.waitKey(0)
