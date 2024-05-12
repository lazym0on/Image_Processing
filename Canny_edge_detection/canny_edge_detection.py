# canny edge detection

from collections import deque

import cv2
import numpy as np
import math


def min_max_scaling(src):
    return 255 * ((src - src.min()) / (src.max() - src.min()))


def get_DoG_filter(fsize, sigma):
    DoG_x = np.zeros((fsize, fsize), np.float64)
    DoG_y = np.zeros((fsize, fsize), np.float64)
    half = fsize // 2
    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
            DoG_x[y + half, x + half] = (-x / (2 * np.pi * sigma ** 4)) * np.exp(
                -((x ** 2 + y ** 2) / (2 * sigma ** 2)))
            DoG_y[y + half, x + half] = (-y / (2 * np.pi * sigma ** 4)) * np.exp(
                -((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    return DoG_x, DoG_y


def calculate_magnitude(gradient_x, gradient_y):
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return magnitude


def non_maximum_suppression(gradient_x, gradient_y, magnitude, n):
    h, w = magnitude.shape
    large_magnitude = np.zeros((h, w))

    for y_k in range(n // 2, h - (n // 2)):
        for x_k in range(n // 2, w - (n // 2)):
            a = math.atan(gradient_y[y_k, x_k] / gradient_x[y_k, x_k])
            max_mag = magnitude[y_k, x_k]
            mag_list = []
            if abs(a) < 1:  # case 1
                for i in range(-(n // 2), (n // 2) + 1):
                    neighbor_mag_pos_y, neighbor_mag_pos_x = y_k + (i * a), x_k + i
                    # 바닥함수를 사용해서 선형 보간값 계산
                    floor_pos_y = np.floor(neighbor_mag_pos_y)
                    ceil_pos_y = np.ceil(neighbor_mag_pos_y)
                    # 거리에 따른 가중치 계산 w1 + w2 = 1.0
                    w1 = abs(neighbor_mag_pos_y - floor_pos_y)
                    w2 = 1 - w1
                    interpolation_value = magnitude[int(floor_pos_y), int(x_k + i)] * w2 + \
                                          magnitude[int(ceil_pos_y), int(x_k + i)] * w1
                    mag_list.append(interpolation_value)
            elif abs(a) > 1:  # case 2
                for i in range(-(n // 2), (n // 2) + 1):
                    neighbor_mag_pos_y, neighbor_mag_pos_x = y_k + i, x_k + (i / a)
                    # 바닥함수를 사용해서 선형 보간값 계산
                    floor_pos_x = np.floor(neighbor_mag_pos_x)
                    ceil_pos_x = np.ceil(neighbor_mag_pos_x)
                    # 거리에 따른 가중치 계산 w1 + w2 = 1.0
                    w1 = abs(neighbor_mag_pos_x - floor_pos_x)
                    w2 = 1 - w1
                    interpolation_value = magnitude[int(y_k + i), int(floor_pos_x)] * w2 + \
                                          magnitude[int(y_k + i), int(ceil_pos_x)] * w1
                    mag_list.append(interpolation_value)
            else:            # case 3
                for i in range(-(n // 2), (n // 2) + 1):
                    neighbor_mag_pos_y, neighbor_mag_pos_x = y_k + i, x_k
                    mag_list.append(magnitude[neighbor_mag_pos_y, neighbor_mag_pos_x])

            # mag_list의 중앙값은 현재 기준이 되는 값이므로 올바른 값 계산을 위해 변경
            mag_list[len(mag_list) // 2] = magnitude[y_k, x_k]
            max_mag = max(mag_list)
            if magnitude[y_k, x_k] >= max_mag:
                large_magnitude[y_k, x_k] = magnitude[y_k, x_k]

    return large_magnitude


def double_thresholding(nms_result, high_threshold, low_threshold):
    h, w = nms_result.shape
    thresholding_result = np.zeros((h, w), dtype=np.int32)

    for row in range(0, h):
        for col in range(0, w):
            if nms_result[row, col] > high_threshold:
                thresholding_result[row, col] = 255
            elif nms_result[row, col] < low_threshold:
                thresholding_result[row, col] = 0
            else:  # low_threshold <= nms_result <= high_threshold
                thresholding_result[row, col] = 128

    return thresholding_result


def determine_edge(thresholding_result):
    # 시작 픽셀값은 weak edge 좌표를 순회하면서 방문하지 않은 좌표
    # weak_edge -> weak edge를 가지는 좌표값을 tuple 형태로 저장
    # visited -> 해당 픽셀 좌표의 방문 여부
    # is_edge -> bfs를 통해 구한 연결 성분에 strong edge가 포함되어 있다면 True로 설정된다.
    visited = np.zeros(thresholding_result.shape, dtype=np.bool_)
    # thresholoing result 배열을 복사
    canny_edge_result = thresholding_result.copy()
    weak_edge = np.where(thresholding_result == 128)

    # length는 weak_edgo[0] 또는 weak_edge[1]에 해당한다.
    for i in range(0, len(weak_edge[0])):
        pos_y = weak_edge[0][i]
        pos_x = weak_edge[1][i]
        if not visited[pos_y, pos_x]:
            is_edge = False
            result = deque([])
            connected_component = []

            result.append([pos_y, pos_x])
            connected_component.append([pos_y, pos_x])
            visited[pos_y, pos_x] = True

            # bfs
            # result는 bfs를 위한 queue이고, connected_component는 순회를 마친 후 연결 성분에 존재하는 element(좌표)
            while result:
                curr_y, curr_x = result.pop()
                for y in range(curr_y - 1, curr_y + 2):
                    for x in range(curr_x - 1, curr_x + 2):
                        if thresholding_result[y, x] >= 128 and not visited[y, x]:
                            # strong edge일 경우 edge임을 결정하기 위해 True로 설정
                            if thresholding_result[y, x] == 255:
                                is_edge = True
                            result.append([y, x])
                            connected_component.append([y, x])
                            visited[y, x] = True

            # weak edge를 strong edge 또는 not edge로 변환
            for j in range(0, len(connected_component)):
                if is_edge:
                    # connected component에 strong edge가 있는 경우
                    edge_y, edge_x = connected_component[j]
                    canny_edge_result[edge_y, edge_x] = 255
                else:
                    # connected component에 strong edge가 없는 경우
                    edge_y, edge_x = connected_component[j]
                    canny_edge_result[edge_y, edge_x] = 0

    return canny_edge_result


def main():
    image = cv2.imread('noise_Lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    DoG_x, DoG_y = get_DoG_filter(fsize=5, sigma=1)
    gradient_y = cv2.filter2D(image, -1, DoG_y)
    gradient_x = cv2.filter2D(image, -1, DoG_x)
    magnitude = calculate_magnitude(gradient_x=gradient_x, gradient_y=gradient_y)
    nms_result = non_maximum_suppression(gradient_x=gradient_x, gradient_y=gradient_y, magnitude=magnitude, n=5)
    thresholding_result = double_thresholding(nms_result=nms_result, high_threshold=10, low_threshold=4)
    canny_edge_result = determine_edge(thresholding_result=thresholding_result)

    cv2.imwrite('magnitude.png', min_max_scaling(magnitude))
    cv2.imwrite('nms.png', min_max_scaling(nms_result))
    cv2.imwrite('thresholding.png', thresholding_result)
    cv2.imwrite('canny_edge.png', canny_edge_result)


if __name__ == '__main__':
    main()
