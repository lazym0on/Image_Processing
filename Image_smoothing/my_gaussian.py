import os
import cv2
import numpy as np
import time
import math


def my_get_Gaussian2D_kernel(ksize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################

    # 2d gaussian kernel 생성

    gaus2D = np.zeros((ksize, ksize))
    s = range(-int(ksize / 2), int(ksize / 2) + 1)
    t = range(-int(ksize / 2), int(ksize / 2) + 1)

    for row in s:
        for col in t:
            numerator = math.exp(-(row**2 + col**2) / 2*sigma**2)
            denominator = (2 * math.pi * sigma**2)
            filter = numerator / denominator
            gaus2D[int(row + (ksize / 2)), int(col + (ksize / 2))] = filter

    return gaus2D


def save_kernel_img(kernel, ksize, sigma):
    kernel_scaled = (kernel - np.min(kernel)) / ((np.max(kernel) - np.min(kernel)) + 1e-10)
    kernel_scaled = (kernel_scaled * 255).astype(np.uint8)  # float -> uint8
    cv2.imwrite(f'gaussian_img_{ksize}_{sigma}.png', kernel_scaled)


def print_kernel(kernel):
    k_h, k_w = kernel.shape

    for row in range(k_h):
        for col in range(k_w):
            print(round(kernel[row, col], 4), end="\t")
        print()


def my_padding(src, pad_shape, pad_type='zeros'):
    (h, w) = src.shape
    p_h, p_w = pad_shape
    pad_img = np.zeros((h + p_h * 2, w + p_w * 2), dtype=np.uint8)
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1:p_w + w]

    else:
        # else is zero padding
        print('zero padding')
        # up
        pad_img[:p_h, p_w:p_w + w] = 0
        # down
        pad_img[p_h + h:, p_w:p_w + w] = 0
        # left
        pad_img[:, :p_w] = 0
        # right
        pad_img[:, p_w + w:] = 0

    return pad_img


def my_filtering(src, kernel, pad_type='zeros'):
    # 과제1 my_filtering 사용
    (h, w) = src.shape
    (k_h, k_w) = kernel.shape

    k_h_half = int(k_h / 2)
    k_w_half = int(k_w / 2)

    # 직접 구현한 my_padding 함수를 이용
    img_pad = my_padding(src, (k_h_half, k_w_half))
    print(f'<img_pad.shape>: {img_pad.shape}')

    dst = np.zeros(src.shape)
    time_start = time.time()
    # filtering 진행하는 반복문 구현
    for row in range(0, h):
        for col in range(0, w):
            filtered_value = 0.
            for s in range(0, k_h):
                for t in range(0, k_w):
                    filtered_value += img_pad[row + s, col + t]
            filtered_value = filtered_value / (k_h * k_w)
            dst[row, col] = filtered_value

    print(f'filtering time: {time.time() - time_start}')

    dst = dst.astype(np.uint8)  # float -> uint8 변환

    return dst


if __name__ == '__main__':
    src = cv2.imread('./Lena.png', cv2.IMREAD_GRAYSCALE)

    kernel_size = 9
    sigma = 1
    gaus2D = my_get_Gaussian2D_kernel(kernel_size, sigma)

    print_kernel(gaus2D)  # kernel 값 출력

    print('2D gaussian filter')
    start = time.time()  # 시간 측정 시작
    dst_gaus2D = my_filtering(src, gaus2D)
    end = time.time()  # 시간 측정 끝
    print('2D time: ', end - start)

    save_kernel_img(gaus2D, kernel_size, sigma)  # kernel 이미지 저장

    cv2.imshow('original', src)
    cv2.imshow(f'2D gaussian img', dst_gaus2D)

    cv2.waitKey()
    cv2.destroyAllWindows()
