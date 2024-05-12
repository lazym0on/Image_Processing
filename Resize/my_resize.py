import cv2
import numpy as np
import math


def match_up_coordinates(old_shape, new_shape):
    (old_h, old_w) = old_shape.shape
    (new_h, new_w) = new_shape
    a_y = (old_h - 1) / (new_h - 1)
    a_x = (old_w - 1) / (new_w - 1)
    b_y = 0.
    b_x = 0.
    return a_y, a_x, b_y, b_x


def my_resize(old_img, new_shape):
    # 빈 배열 만들기
    new_img = np.zeros(new_shape)
    h, w = new_img.shape
    old_h, old_w = old_img.shape

    # match_up_coordinates
    a_y, a_x, b_y, b_x = match_up_coordinates(old_img, new_shape)

    # new_img의 모든 픽셀 값 채워넣기
    for row in range(h):
        for col in range(w):
            # y, x 좌표 매칭
            y = a_y * row
            x = a_x * col

            # ⌊𝑦⌋, ⌊𝑥⌋, ⌊𝑦+1⌋, ⌊𝑥+1⌋ 구하기
            y_floor = math.floor(y)
            x_floor = math.floor(x)
            y_ceil = min(math.floor(y + 1), old_h - 1)
            x_ceil = min(math.floor(x + 1), old_w - 1)

            # binary interpolation을 통해 픽셀 값 구하기
            intensity =\
                (y_ceil - y)*(x_ceil - x)*old_img[y_floor, x_floor] +\
                (y_ceil - y)*(x - x_floor)*old_img[y_floor, x_ceil] +\
                (y - y_floor)*(x_ceil - x)*old_img[y_ceil, x_floor] +\
                (y - y_floor)*(x - x_floor)*old_img[y_ceil, x_ceil]

            new_img[row, col] = round(intensity)

    return new_img


def main():
    old_img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    new_img_1000by1024 = my_resize(old_img, new_shape=(1000, 1024))
    new_img_256by200 = my_resize(old_img, new_shape=(256, 200))
    cv2.imwrite('new_img_1000by1024.png', new_img_1000by1024)
    cv2.imwrite('new_img_256by200.png', new_img_256by200)

    return


if __name__ == '__main__':
    main()
