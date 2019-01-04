import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

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


def houghLines(grad_img):
    """
    从梯度图进行hough变换，检测直线
    :param grad_img: 梯度图
    :return: 检测出来的直线的极坐标表示
    """
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
    valid_lines = np.zeros((top_k, 2))
    exist_num = 0
    for i in range(0, m * n):
        row_ind, col_ind = tuple(argmax_ind[i])
        theta = theta_range[row_ind]
        rho = rho_range[col_ind]
        if is_new_line(theta, rho, valid_lines, exist_num):
            # 遇到新的线了
            # print(theta,rho)
            valid_lines[exist_num][0] = theta
            valid_lines[exist_num][1] = rho
            exist_num += 1
            if exist_num >= top_k:
                break
    # ----------------------------------过滤 2: 倾角在45度也不予考虑-------------------------------
    # valid_angle = np.abs(valid_data[:, 0] - 0.785) > 0.2
    # valid_data = valid_data[valid_angle, :]
    #
    # valid_angle = valid_data[:, 0] > 0
    # valid_data = valid_data[valid_angle, :]
    return valid_lines


def detect_corners(gray_img):
    grad_img = get_grad_img(gray_img)
    polar_lines = houghLines(grad_img)
    # -------------------------------计算交点----------------------------------
    # 1. 为了化简计算,把直线分成接近水平/垂直, 两种直线
    vert_ind = np.abs(polar_lines[:, 0] - 1.5) > 0.5
    vert_lines = polar_lines[vert_ind, :]  # 接近垂直的直线
    hori_lines = polar_lines[np.logical_not(vert_ind), :]  # 接近水平的直线

    # 排序: 为了能够组成正方形,先进行排序
    test = np.argsort(np.abs(vert_lines[:, 1]))
    vert_lines = vert_lines[test, :]

    test = np.argsort(np.abs(hori_lines[:, 1]))
    hori_lines = hori_lines[test, :]

    # 2. 计算交点
    points = []
    num_vert_lines = vert_lines.shape[0]
    num_hori_lines = hori_lines.shape[0]
    for i in range(num_vert_lines):
        for j in range(num_hori_lines):
            point = get_intersection_points(vert_lines[i], hori_lines[j])
            points.append([point[1], point[0]])
            # cv2.circle(grad_img, tuple(point), 10, (255, 0, 0), 2)  # 画出交点

    # 3. 近似面积最大的为角点
    points = np.array(points).reshape(num_vert_lines, num_hori_lines, 2)
    max_area = 0
    for i in range(num_vert_lines - 1):
        for j in range(num_hori_lines - 1):
            left_top = points[i][j]
            left_bottom = points[i][j + 1]
            right_top = points[i + 1][j]
            right_bottom = points[i + 1][j + 1]
            area = get_approx_area(left_top, left_bottom, right_top, right_bottom)
            if area > max_area:
                max_area = area
                point_seq = (left_top, right_top, right_bottom, left_bottom)
    # 绘制检测到的直线
    for i in range(polar_lines.shape[0]):
        theta, rho = tuple(polar_lines[i])
        # print(theta, rho)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(grad_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return grad_img, np.array(point_seq)


def get_approx_area(p1, p2, p3, p4):
    top_line = np.abs(
        p1[1] - p3[1]
    )
    bottem_line = np.abs(
        p2[1] - p4[1]
    )
    left_line = np.abs(
        p1[0] - p2[0]
    )
    right_line = np.abs(
        p3[0] - p4[0]
    )
    return (top_line + bottem_line) * (left_line + right_line)


def is_new_line(theta, rho, valid_data, exist_num):
    for i in range(exist_num):
        theta = 0 if theta - 3.1 > 0 else theta  # 角度3.1...和零度是一样的
        if theta - valid_data[i][0] < 0.2 and np.square(np.abs(rho) - np.abs(valid_data[i][1])) < 1000:
            # 角度相近 & rho相近
            return False
    return True


def get_intersection_points(line1, line2):
    """
    由极坐标表示的line1, line2,求出角点(矩阵求解方程)
    :param line1: [theta1, rho1]
    :param line2: [theta2,rho2]
    :return: row, col
    """
    rho_mat = np.array(
        [line1[1], line2[1]]
    )
    theta_mat = np.array(

        [[np.cos(line1[0]), np.sin(line1[0])],
         [np.cos(line2[0]), np.sin(line2[0])]]
    )
    inv_theta_mat = np.linalg.inv(theta_mat)
    result = np.matmul(inv_theta_mat, rho_mat).astype(np.int32)
    return result.astype(np.int32)  # 由于是坐标,需要改成int


def harries(gray):
    # corners = cv2.cornerHarris(img, 2, 3, 0.04)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(gray, (x, y), 3, 255, -1)
    return gray


if __name__ == "__main__":
    # path = "./data/000026.jpg"
    # path = './data/000872.jpg'
    # path = './data/001201.jpg'
    # path = './data/001402.jpg'
    # path = './data/001552.jpg'
    path = "./data/1.jpg"
    img = cv2.imread(path)
    img = cv2.resize(img, (504, 738))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect_img, point_seq = detect_corners(gray_img)
    print(np.array(point_seq))
    plt.imshow(detect_img)
    plt.show()
    # cv2.imshow('dst', detect_img)
    # cv2.waitKey(0)
