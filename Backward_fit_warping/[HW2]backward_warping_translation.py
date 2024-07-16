import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_out_list(output_list=[], titles=[], figsize=(10, 10)):
    plt.rcParams['figure.figsize'] = figsize
    row = 1
    col = len(output_list)

    for i in range(len(output_list)):
        image_index = i + 1
        plt.subplot(row, col, image_index)
        plt.imshow(output_list[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
    plt.show()


def backward_fit(src, matrix):
    h, w = src.shape
    src = src.astype(np.float32)
    M_inv = np.linalg.inv(matrix)

    # 변환 좌표에 따른 이미지 크기 계산
    left_top_pos = np.dot(matrix, np.array([0, 0, 1]))  # x1_prime, y1_prime
    left_bottom_pos = np.dot(matrix, np.array([0, h - 1, 1]))  # x3_prime, y3_prime
    right_top_pos = np.dot(matrix, np.array([w - 1, 0, 1]))  # x2_prime, y2_prime
    right_bottom_pos = np.dot(matrix, np.array([w - 1, h - 1, 1]))  # x4_prime, y4_prime

    # 변환 전 원본 이미지의 좌표 계산
    x1, y1 = 0, 0
    x2, y2 = w - 1, 0
    x3, y3 = 0, h - 1
    x4, y4 = w - 1, h - 1

    y_max = max(y1, y3, left_top_pos[1], left_bottom_pos[1])
    y_min = min(y1, y3, left_top_pos[1], left_bottom_pos[1])
    x_max = max(x1, x2, left_top_pos[0], right_bottom_pos[0])
    x_min = min(x1, x2, left_top_pos[0], right_bottom_pos[0])

    H_new = int(np.round(y_max - y_min))
    W_new = int(np.round(x_max - x_min))

    dst = np.zeros((H_new, W_new))

    # 변환된 좌표에서 x, y가 가장 작은 지점이 backward warping의 시작 지점
    start_x = x_min
    start_y = y_min

    # backward warping
    for row in range(H_new):
        for col in range(W_new):
            pos = np.dot(M_inv, np.array([start_x + col, start_y + row, 1]))
            pos_x = pos[0]
            pos_y = pos[1]
            intensity = 0

            # 입력 영상의 범위를 벗어나지 않아야함.
            if 0 <= pos_y < h and 0 <= pos_x < w:
                y_floor = int(pos_y)
                x_floor = int(pos_x)
                y_ceil = min(int(np.ceil(pos_y)), h - 1)
                x_ceil = min(int(np.ceil(pos_x)), w - 1)

                # 만약 역변환된 좌표 x, y 모두 입력 영상의 픽셀(정수 좌표)으로 나타날 경우
                # 해당 src 픽셀값 그대로 사용
                if x_floor == x_ceil and y_floor == y_ceil:
                    intensity = src[y_floor, x_floor]
                    dst[row, col] = intensity
                    continue

                # y가 두 점 사이에 위치한 경우
                # linear interpolation
                if x_floor == x_ceil:
                    x = x_floor
                    intensity = (y_ceil - pos_y) * src[y_floor, x] + \
                                (pos_y - y_floor) * src[y_ceil, x]
                    dst[row, col] = intensity
                    continue

                # x가 두 점 사이에 위치한 경우
                # linear interpolation
                if y_floor == y_ceil:
                    y = y_floor
                    intensity = (x_ceil - pos_x) * src[y, x_floor] + \
                                (pos_x - x_floor) * src[y, x_ceil]
                    dst[row, col] = intensity
                    continue

                # 입력 영상의 x, y 좌표가 모두 소수로 나타날 경우 (네 점 사이에 위치)
                # bilinear interpolation
                intensity = (y_ceil - pos_y) * (x_ceil - pos_x) * src[y_floor, x_floor] + \
                            (y_ceil - pos_y) * (pos_x - x_floor) * src[y_floor, x_ceil] + \
                            (pos_y - y_floor) * (x_ceil - pos_x) * src[y_ceil, x_floor] + \
                            (pos_y - y_floor) * (pos_x - x_floor) * src[y_ceil, x_ceil]
                dst[row, col] = intensity

    dst = np.clip(np.round(dst), 0, 255).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # translation
    M1 = np.array([[1, 0, 500],
                   [0, 1, 600],
                   [0, 0, 1]])
    M2 = np.array([[1, 0, -500],
                   [0, 1, 600],
                   [0, 0, 1]])
    M3 = np.array([[1, 0, 500],
                   [0, 1, -600],
                   [0, 0, 1]])
    M4 = np.array([[1, 0, -500],
                   [0, 1, -600],
                   [0, 0, 1]])

    final1 = backward_fit(src, M1)
    final2 = backward_fit(src, M2)
    final3 = backward_fit(src, M3)
    final4 = backward_fit(src, M4)

    plot_out_list([src, final1, final2, final3, final4], ['Original', 'final1', 'final2', 'final3', 'final4'],
                  figsize=(15, 15))

    cv2.waitKey()
    cv2.destroyAllWindows()
