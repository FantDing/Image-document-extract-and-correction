import os
import cv2
from transform import cvtColor, warpAffine
from corner_detection import detect_corners

if __name__ == "__main__":
    # filename = "000026.jpg"
    # filename = "000029.jpg"
    filename= '000872.jpg'
    # filename= '001201.jpg'
    # filename= '001402.jpg'
    # filename = '001552.jpg'
    path = os.path.join('./data', filename)
    src_img = cv2.imread(path)
    gray_img = cvtColor(src_img)
    # 检测角点：从左上角开始，顺时针
    detect_img, detected_corner = detect_corners(gray_img)
    # print(detect_img.shape)
    # print(detected_corner)
    tar_img = warpAffine(src_img, detected_corner, (504, 378))
    cv2.imwrite(os.path.join('./result', filename), tar_img)
