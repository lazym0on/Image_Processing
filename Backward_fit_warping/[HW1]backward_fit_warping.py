import cv2
import numpy as np
import matplotlib.pyplot as plt


def backward_fit(src, matrix):
    h, w = src.shape
    src = src.astype(np.float32)
    # matrix 역행렬 구하기
    M_inv = np.linalg.inv(matrix)

    # 변환 좌표에 따른 이미지 크기 계산
    left_top_pos = np.dot(matrix, np.array([0, 0, 1]))
    left_bottom_pos = np.dot(matrix, np.array([0, h - 1, 1]))
    right_top_pos = np.dot(matrix, np.array([w - 1, 0, 1]))
    right_bottom_pos = np.dot(matrix, np.array([w - 1, h - 1, 1]))

    y_max = max(left_top_pos[1], left_bottom_pos[1], right_top_pos[1], right_bottom_pos[1])
    y_min = min(left_top_pos[1], left_bottom_pos[1], right_top_pos[1], right_bottom_pos[1])
    x_max = max(left_top_pos[0], left_bottom_pos[0], right_top_pos[0], right_bottom_pos[0])
    x_min = min(left_top_pos[0], left_bottom_pos[0], right_top_pos[0], right_bottom_pos[0])

    H_new = np.round(y_max - y_min).astype(np.int32)
    W_new = np.round(x_max - x_min).astype(np.int32)

    dst = np.zeros((H_new, W_new))

    # 변환된 좌표에서 원점 구하기
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
    src = cv2.imread('./Lena.png', cv2.IMREAD_GRAYSCALE)

    # Rotation 20 -> shearing -> scaling
    M_ro = np.array([[np.cos(np.deg2rad(20)), -np.sin(np.deg2rad(20)), 0],
                     [np.sin(np.deg2rad(20)), np.cos(np.deg2rad(20)), 0],
                     [0, 0, 1]])

    M_sh = np.array([[1, 0.2, 0],
                     [0, 1, 0],
                     [0, 0, 1]])

    M_sc = np.array([[1.2, 0, 0],
                     [0, 1.2, 0],
                     [0, 0, 1]])

    # hole 현상 확인용
    # M_sc = np.array([[  4, 0, 0],
    #                  [  0, 4, 0],
    #                  [  0, 0, 1]])

    #M = M_sc

    # 여러 변환을 하나의 행렬로 만들기 위해 행렬곱을 할땐 순서가 반대
    M = M_sc @ M_sh @ M_ro

    final = backward_fit(src, M)

    cv2.imshow('lena_gray', src)
    cv2.imshow('final', final)
    cv2.imwrite('[HW1]final.png', final)

    cv2.waitKey()
    cv2.destroyAllWindows()
