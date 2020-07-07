import numpy as np


class Conv2d:
    '''
    使用矩阵乘法实现卷积操作
    '''

    def __init__(self, k, s, c_in, c_out, mode='same', weight=None):
        self.k, self.s, self.c_in, self.c_out, self.mode = k, s, c_in, c_out, mode
        if weight is None:
            self.weight = np.random.random((c_in, k, k, c_out))
        else:
            self.weight = weight
        assert mode in ['same', 'full', 'valid']

    def __call__(self, image):
        assert image.shape[
                   0] == self.c_in, f"image in channel {image.shape[0]} not equal to weight channel {self.weight.shape[0]}"
        c, h, w = image.shape
        if self.mode == "same":
            p_h = (self.s * (h - 1) + self.k - h) // 2
            p_w = (self.s * (w - 1) + self.k - w) // 2
        elif self.mode == 'valid':
            p_h, p_w = 0, 0
        elif self.mode == 'full':
            p_h, p_w = self.k - 1, self.k - 1
        else:
            assert False, 'error mode'
        out_h = (h + 2 * p_h - self.k) // self.s + 1
        out_w = (w + 2 * p_w - self.k) // self.s + 1

        # 填充后的image
        padded_img = np.zeros((c, h + 2 * p_h, w + 2 * p_w))
        padded_img[:, p_h:p_h + h, p_w:p_w + w] = image
        # image_mat
        image_mat = np.zeros((out_h * out_w, self.k * self.k * c))
        row = 0
        for i in range(out_h):
            for j in range(out_w):
                window = padded_img[:, i * self.s:(i * self.s + self.k), j * self.s:(j * self.s + self.k)]
                image_mat[row] = window.flatten()
                row += 1
        # 矩阵乘法
        res = np.dot(image_mat, self.weight.reshape(-1, self.c_out))
        res = np.reshape(res, (out_h, out_w, self.c_out))
        return np.transpose(res, (2, 0, 1))


if __name__ == "__main__":
    image = np.ones((3, 10, 10))
    conv2d = Conv2d(3, 1, 3, 16, 'valid')
    res = conv2d(image)
    print(res.shape)
