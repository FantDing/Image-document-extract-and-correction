import numpy as np
import math
import cv2
from matplotlib import pyplot as plt


def build_equ(four_corners):
    X = np.zeros(shape=(4, 4))
    for i, (x0, y0) in enumerate(four_corners):
        X[i, :] = [x0, y0, x0 * y0, 1]
    return X


if __name__ == "__main__":
    # 图片角点顺序如下：
    # 0 1
    # 3 2
    # src_img_path="./data/000026.jpg"
    # detected_corner = [[87,96], [99,330], [459,339], [440,57]]

    src_img_path="./data/000872.jpg"
    detected_corner=[[119,74],[129,296],[442,306],[444,55]]

    detected_corner = np.array(detected_corner, dtype=np.int32)

    height, width = [504, 378]
    target_corner = np.array([[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]], dtype=np.int32)

    # 计算x,y变化矩阵T_x,T_y
    leftMatrix = build_equ(target_corner)
    inversed_leftMat = np.linalg.inv(leftMatrix)
    X1 = detected_corner[:, 0]
    T_x = np.matmul(inversed_leftMat, X1)

    Y1 = detected_corner[:, 1]
    T_y = np.matmul(inversed_leftMat, Y1)

    # test = np.matmul(leftMatrix[2, :], T_y)
    # print(test)

    # 双线性插值
    src_img = cv2.imread(src_img_path)
    # pick_img=cv2.copyMakeBorder(src_img,0,1,0,1,cv2.BORDER_REPLICATE)
    tar_img = np.zeros(shape=(height, width, 3))
    # 遍历每个像素，进行后向插值
    for channel in range(3):
        for i in range(height):
            for j in range(width):
                equation_coefficient = np.array([i, j, i * j, 1])
                x = np.matmul(equation_coefficient, T_x)
                y = np.matmul(equation_coefficient, T_y)
                x_ceil = math.ceil(x)
                x_floor = math.floor(x)
                y_ceil = math.ceil(y)
                y_floor = math.floor(y)
                p1 = x - x_floor
                p2 = 1 - p1
                q1 = y - y_floor
                q2 = 1 - q1

                try:
                    tar_img[i, j, channel] = p2 * q2 * src_img[x_floor, y_floor, channel] + \
                                             src_img[x_ceil, y_floor, channel] * p1 * q2 + \
                                             src_img[x_ceil, y_ceil, channel] * q1 * p1 + \
                                             src_img[x_floor, y_ceil, channel] * q1 * p2
                except:
                    print(x_ceil,x_floor,y_ceil,y_floor)
                    print(src_img.shape)
                    assert 1==9
    cv2.imshow('Window', np.uint8(tar_img))
    cv2.waitKey(0)
    # plt.imshow(np.uint8(tar_img))
    # plt.show()
