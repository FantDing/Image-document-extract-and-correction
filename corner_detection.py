import cv2
import numpy as np
import math
from scipy.signal import fftconvolve as conv2d

'''
可调参数
@get_grad_img: grad阈值
'''


def get_grad_img(gray_img):
    gauss = np.ones(shape=(3, 3)) / 6
    smoothed_img = cv2.filter2D(gray_img, -1, kernel=gauss)
    laplace = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )
    grad_img = cv2.filter2D(smoothed_img, -1, kernel=laplace)
    grad_img = np.where(grad_img > 30, grad_img, 0)
    return grad_img


def hough_transform(gray_img):
    grad_img = get_grad_img(gray_img)
    # --------------------------------------投票------------------------------------------
    rho_max = math.sqrt(grad_img.shape[0] * grad_img.shape[0] + grad_img.shape[1] * grad_img.shape[1])
    m, n = 180, 2000
    theta_range = np.linspace(0, np.pi, m)
    rho_range = np.linspace(-rho_max, rho_max, n)
    # 投票的表格
    vote_table = np.zeros(shape=(m, n))

    row_cor, col_cor = np.where(grad_img > 0)  # 挑出有选举权的点,假设有K个
    cor_mat = np.stack((row_cor, col_cor), axis=1)  # K*2
    K = cor_mat.shape[0]

    cos_theta = np.cos(theta_range)
    sin_theta = np.sin(theta_range)
    # 这是一个大坑，row实际对应的是y
    # theta_mat = np.stack((cos_theta, sin_theta), axis=0)  # 2*m
    theta_mat = np.stack((sin_theta, cos_theta), axis=0)  # 2*m

    y_mat = np.matmul(cor_mat, theta_mat)  # K*m

    rho_ind = (
            (y_mat - (-rho_max)) * (n - 1) / (rho_max - (-rho_max))
    ).astype(np.int32)  # K*m
    rho_ind = np.ravel(rho_ind, order='F')  # 在列方向stack

    theta_ind = np.arange(0, m)[:, np.newaxis]
    theta_ind = np.repeat(theta_ind, K)

    np.add.at(vote_table, (theta_ind, rho_ind), 1)  # 在vote_table中投票
    # ----------------------------------过滤 1： 选出不同的直线-------------------------------
    # 取出top_k条不同的直线
    top_k = 5
    argmax_ind = np.dstack(np.unravel_index(np.argsort(-vote_table.ravel(), ), (m, n)))
    argmax_ind = argmax_ind[0, :, :]
    valid_data = np.zeros((top_k, 2))
    exist_num = 0
    for i in range(0, m * n):
        row_ind, col_ind = tuple(argmax_ind[i])
        theta = theta_range[row_ind]
        rho = rho_range[col_ind]
        if is_new_line(theta, rho, valid_data, exist_num):
            # 遇到新的线了
            valid_data[exist_num][0] = theta
            valid_data[exist_num][1] = rho
            exist_num += 1
            if exist_num >= top_k:
                break


    #  绘制检测到的直线
    for i in range(top_k):
        theta, rho = tuple(valid_data[i])
        print(theta, rho)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(grad_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return grad_img


def is_new_line(theta, rho, valid_data, exist_num):
    for i in range(exist_num):
        theta = 0 if theta - 3.1 > 0 else theta
        if theta - valid_data[i][0] < 0.2 and np.square(np.abs(rho) - np.abs(valid_data[i][1])) < 1000:
            # 角度相近 & rho相近
            return False
    return True


def harries(gray):
    # corners = cv2.cornerHarris(img, 2, 3, 0.04)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(gray, (x, y), 3, 255, -1)
    return gray


if __name__ == "__main__":
    path = "./data/000026.jpg"
    # path='./data/000872.jpg'
    # path = './data/001201.jpg'
    # path = './data/001402.jpg'
    # path = './data/001552.jpg'
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detect_img = hough_transform(gray_img)
    print(detect_img.shape)
    cv2.imshow('dst', detect_img)
    cv2.waitKey(0)
