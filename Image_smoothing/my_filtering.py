import cv2
import numpy as np
import time
def my_padding(src, pad_size, pad_type='zeros'):
    (h, w) = src.shape
    p_h, p_w = pad_size
    pad_img = np.zeros((h + p_h * 2, w + p_w * 2), dtype=np.uint8)
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        #down
        pad_img[p_h + h:, p_w:p_w + w] = src[h-1, :]
        #left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        #right
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

    print(f'filtering time: {time.time()-time_start}')

    dst = dst.astype(np.uint8) # float -> uint8 변환

    return dst

if __name__ == '__main__':
    src = cv2.imread('./Lena_noise.png', cv2.IMREAD_GRAYSCALE)
    
    # average filter 생성
    kernel = np.ones((5, 5))
    kernel = kernel / np.sum(kernel)
    print('<kernel>')
    print(kernel)

    dst = my_filtering(src, kernel)

    print(f'src.shape: {src.shape}')
    print(f'dst.shape: {dst.shape}')

    cv2.imshow('original', src)
    cv2.imshow('dst', dst)

    #
    cv2.imwrite('average_filter.png', dst)
    #

    cv2.waitKey()
    cv2.destroyAllWindows()