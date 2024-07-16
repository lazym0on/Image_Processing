# image segmentation

import cv2
import numpy as np
from collections import deque


def get_hist(src):
    h, w = src.shape
    hist = np.zeros(256, dtype=np.uint32)
    for row in range(h):
        for col in range(w):
            intensity = src[row, col]
            hist[intensity] += 1
    return hist


def otsu_method(src):
    # mask_size는 mask 이미지에서 255값을 가지는 픽셀의 수
    h, w = src.shape

    # get_hist 함수 사용
    hist = get_hist(src)

    p = hist / (h * w)

    # k가 0일때 q, m, var 계산
    k = 0
    q1 = p[0]
    m1 = 0
    m = sum(i * p[i] for i in range(len(hist)))
    m2 = m

    between = []
    var = (q1 * (1 - q1)) * (m1 - m2) * (m1 - m2)
    between.append(var)

    # moving average를 사용하여 inter-class variance 계산
    for i in range(1, len(hist)):
        q1_n = q1 + p[i]
        m1_n = ((q1 * m1) + (i * p[i])) / (q1_n + 1e-10)
        m2_n = (m - (q1_n * m1_n)) / (1 - q1_n + 1e-10)

        m1 = m1_n
        m2 = m2_n
        q1 = q1_n

        var = q1 * (1 - q1) * (m1 - m2) * (m1 - m2)
        between.append(var)

    # var의 최댓값 지점의 k 탐색
    max_var = between[0]
    for i in range(1, len(between)):
        if max_var < between[i]:
            max_var = between[i]
            k = i

    print('threshold k : ', k)
    # k를 기준으로 thresholding
    binary = src.copy()

    for row in range(0, h):
        for col in range(0, w):
            if binary[row, col] > k:
                binary[row, col] = 0
            else:
                binary[row, col] = 255

    return binary


def dilation(B, S):
    dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    dx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]

    h, w = B.shape
    s_h, s_w = S.shape
    # B에서 255의 값을 가지는 픽셀을 -1로 치환하여 복사
    dst = np.zeros((h, w), np.float32)
    dst[B == 255] = -1

    # B에서 1로 된 영역의 좌표를 튜플로 받음
    obj = np.where(B == 255)

    queue = deque()
    for i in range(len(obj[0])):
        now_y, now_x = obj[0][i], obj[1][i]
        if dst[now_y][now_x] != -1:
            continue
        queue.append((now_y, now_x))
        connect = [(now_y, now_x)]
        # bfs
        while queue:
            now_y, now_x = queue.popleft()
            for j in range(s_h * s_w):
                next_y = now_y + dy[j]
                next_x = now_x + dx[j]
                if 0 <= next_y < h and 0 <= next_x < w:
                    if dst[next_y][next_x] == -1:
                        if (next_y, next_x) not in connect:
                            queue.append((next_y, next_x))
                            connect.append((next_y, next_x))
        while connect:
            connect_y, connect_x = connect.pop()
            for j in range(s_h * s_w):
                y_dilation = connect_y + dy[j]
                x_dilation = connect_x + dx[j]
                S_value = S[dy[j], dx[j]]
                if 0 <= y_dilation < h and 0 <= x_dilation < w:
                    if S_value or (dst[y_dilation][x_dilation] == -1):
                        dst[y_dilation][x_dilation] = 1
    dst *= 255
    return dst


def erosion(B, S):
    dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    dx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]

    h, w = B.shape
    s_h, s_w = S.shape
    # B에서 255의 값을 가지는 픽셀을 -1로 치환하여 복사
    dst = np.zeros((h, w), np.float32)
    dst[B == 255] = -1

    # B에서 1로 된 영역의 좌표를 튜플로 받음
    obj = np.where(B == 255)

    queue = deque()
    for i in range(len(obj[0])):
        now_y, now_x = obj[0][i], obj[1][i]
        if dst[now_y][now_x] != -1:
            continue
        queue.append((now_y, now_x))
        connect = [(now_y, now_x)]
        # bfs
        while queue:
            now_y, now_x = queue.popleft()
            for j in range(s_h * s_w):
                next_y = now_y + dy[j]
                next_x = now_x + dx[j]
                if 0 <= next_y < h and 0 <= next_x < w:
                    if dst[next_y][next_x] == -1:
                        if (next_y, next_x) not in connect:
                            queue.append((next_y, next_x))
                            connect.append((next_y, next_x))
        while connect:
            connect_y, connect_x = connect.pop()
            for j in range(s_h * s_w):
                y_erosion = connect_y + dy[j]
                x_erosion = connect_x + dx[j]
                S_value = S[dy[j], dx[j]]
                if 0 <= y_erosion < h and 0 <= x_erosion < w:
                    if S_value and (B[y_erosion][x_erosion] == 0):
                        dst[connect_y][connect_x] = 0
                        break
                if j == s_h * s_w - 1:
                    dst[connect_y][connect_x] = 1
    dst *= 255
    return dst


def opening(B, S):
    B_erosion = erosion(B, S)
    dst = dilation(B_erosion, S)
    return dst


def closing(B, S):
    B_dilation = dilation(B, S)
    dst = erosion(B_dilation, S)
    return dst


def main():
    # 1. 흑백 이미지 불러오기
    original = cv2.imread('../../ip/Image_segmentation/cell.png')
    gray_scale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 2. otsu's method를 적용하여 Binary image로 만들기
    otsu_binary_image = otsu_method(gray_scale)
    cv2.imwrite('./cell_otsu.png', otsu_binary_image)

    B_test = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])

    # 3. Morphology 연산을 적용하여 noise를 제거한 Mask 생성

    # dilation(B_test * 255, S)
    # erosion(B_test * 255, S)
    # opening(B_test * 255, S)
    # closing(B_test * 255, S)

    # dilation
    # mask = dilation(otsu_binary_image, S)
    # cv2.imwrite('./cell_otsu_dilation.png', mask)

    # erosion
    # mask = erosion(otsu_binary_image, S)
    # cv2.imwrite('./cell_otsu_erosion.png', mask)

    # opening
    # mask = opening(mask, S)
    # cv2.imwrite('./cell_otsu_opening.png', mask)

    # closing
    # mask = closing(otsu_binary_image, S)
    # cv2.imwrite('./cell_otsu_closing.png', mask)

    # 1. opening / 2. closing
    # mask = opening(otsu_binary_image, S)
    # mask = closing(mask, S)
    # cv2.imwrite('./cell_otsu_open_close.png', mask)

    # 1. closing / 2.opening
    # mask = closing(otsu_binary_image, S)
    # mask = opening(mask, S)
    # cv2.imwrite('./cell_otsu_close_open.png', mask)

    # 1. opening / 2. dilation
    mask = opening(otsu_binary_image, S)
    mask = dilation(mask, S)
    cv2.imwrite('./cell_otsu_open_dilation.png', mask)

    # 4. 원본 이미지에 mask 씌우기
    original[mask == 0] -= 127
    cv2.imwrite('result.png', original)


if __name__ == '__main__':
    main()
