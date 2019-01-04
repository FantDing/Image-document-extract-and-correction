import numpy as np
import math
import cv2
from corner_detection import detect_corners


def build_equ(four_corners):
    X = np.zeros(shape=(4, 4))
    for i, (x0, y0) in enumerate(four_corners):
        X[i, :] = [x0, y0, x0 * y0, 1]
    return X


def warpAffine(src_img,detected_corner):
    height, width = [504, 378]
    target_corner = np.array([[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]], dtype=np.int32)
    # 计算x,y变化矩阵T_x,T_y
    leftMatrix = build_equ(target_corner)
    inversed_leftMat = np.linalg.inv(leftMatrix)
    X1 = detected_corner[:, 0]
    T_x = np.matmul(inversed_leftMat, X1)

    Y1 = detected_corner[:, 1]
    T_y = np.matmul(inversed_leftMat, Y1)

    tar_img = fast_bi_inter(src_img, height, width, T_x, T_y)
    return tar_img


def fast_bi_inter(src_img, height, width, T_x, T_y):
    """
    使用矩阵计算，实现快速双线性插值
    :param src_img:
    :param height:
    :param width:
    :return:
    """
    row_same = np.arange(0, height)
    row_same = row_same[:, np.newaxis]
    row_same = np.tile(row_same, (1, width))

    col_same = np.arange(0, width)
    col_same = col_same[np.newaxis, :]
    col_same = np.tile(col_same, (height, 1))

    mix_para = row_same * col_same

    ones_para = np.ones(shape=(height, width))

    paras = np.stack((row_same, col_same, mix_para, ones_para), axis=2)

    paras = np.reshape(paras, (height * width, 4))

    # 找到原图中的坐标
    x = np.matmul(paras, T_x)
    y = np.matmul(paras, T_y)
    # 确定p1,p2,q1,q2
    x_ceil = np.ceil(x).astype(int)
    x_floor = np.floor(x).astype(int)

    y_ceil = np.ceil(y).astype(int)
    y_floor = np.floor(y).astype(int)

    p1 = x - x_floor
    # print(p1.shape) #(190512,)
    p2 = 1 - p1
    q1 = y - y_floor
    q2 = 1 - q1

    tar_img = np.expand_dims(p2 * q2, axis=1) * src_img[x_floor, y_floor, :] + \
              np.expand_dims(p1 * q2, axis=1) * src_img[x_ceil, y_floor, :] + \
              np.expand_dims(q1 * p1, axis=1) * src_img[x_ceil, y_ceil, :] + \
              np.expand_dims(q1 * p2, axis=1) * src_img[x_floor, y_ceil, :]

    tar_img = np.reshape(tar_img, (height, width, 3))
    return tar_img


def main():
    # 图片角点顺序如下：
    # 0 1
    # 3 2
    # src_img_path="./data/000026.jpg"
    # detected_corner = [[87,96], [99,330], [459,339], [440,57]]

    src_img_path = "./data/000872.jpg"
    detected_corner = [[119, 74], [129, 296], [442, 306], [444, 55]]

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
                    print(x_ceil, x_floor, y_ceil, y_floor)
                    print(src_img.shape)
                    assert 1 == 9
    cv2.imshow('Window', np.uint8(tar_img))
    cv2.waitKey(0)


def cvtColor(src_img):
    """
    转彩色图像为灰度图像。模仿opencv命名
    :param src_img: 彩色图像
    :return: 灰度图像
    """
    gray_img = np.sum(src_img, axis=2) / 3
    return gray_img.astype(np.uint8)


if __name__ == "__main__":
    # path = "./data/1.jpg"
    path = "./data/000026.jpg"
    # path = './data/000872.jpg'
    # path = './data/001201.jpg'
    # path = './data/001402.jpg'
    # path = './data/001552.jpg'
    src_img = cv2.imread(path)
    gray_img = cvtColor(src_img)
    detect_img, detected_corner = detect_corners(gray_img)

    tar_img=warpAffine(src_img)
    cv2.imshow('Window', np.uint8(tar_img))
    cv2.waitKey(0)
    cv2.imshow('dst', detect_img)
    cv2.waitKey(0)
