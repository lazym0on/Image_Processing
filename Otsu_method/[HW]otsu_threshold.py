import cv2
import numpy as np


def otsu_method_by_inter_class_variance(src, mask):
    # mask_size는 mask 이미지에서 255값을 가지는 픽셀의 수
    h, w = src.shape
    hist_mask = np.zeros(256, dtype=np.uint32)
    mask_pos = []

    # mask에 해당하는 영역의 histogram 생성
    for row in range(0, h):
        for col in range(0, w):
            if mask[row, col] == 255:
                mask_pos.append((row, col))
                intensity = src[row, col]
                hist_mask[intensity] += 1

    # print(sum(hist_mask))
    mask_size = len(mask_pos)
    p = hist_mask / mask_size
    # print(sum(p))

    # k가 0일때 q, m, var 계산
    k = 0
    q1 = p[0]
    m1 = 0
    m = 0
    for i in range(1, len(hist_mask)):
        m += i * p[i]

    m2 = m

    between = []
    var = (q1 * (1 - q1)) * (m1 - m2) * (m1 - m2)
    between.append(var)

    # moving average를 사용하여 inter-class variance 계산
    for i in range(1, len(hist_mask)):
        q1_n = q1 + p[i]
        m1_n = ((q1 * m1) + (i * p[i])) / (q1_n + 1e-12)
        m2_n = (m - (q1_n * m1_n)) / (1 - q1_n + 1e-12)
        m1 = m1_n
        m2 = m2_n
        q1 = q1_n
        var = (q1 * (1 - q1)) * (m1 - m2) * (m1 - m2)
        if var < 0:
            var = 0.
        between.append(var)

    # var의 최댓값 지점의 k 탐색
    max_var = between[0]
    for i in range(1, len(between)):
        if max_var < between[i]:
            max_var = between[i]
            k = i

    # k를 기준으로 thresholding
    fat = np.zeros((h, w), np.uint8)

    # fat_size는 mask 영역 내의 src 픽셀값이 임계값 k보다 높거나 같은 경우 count
    fat_size = 0
    for i in range(0, len(mask_pos)):
        if src[mask_pos[i]] >= k:
            fat_size += 1
            fat[mask_pos[i]] = 255
        else:
            fat[mask_pos[i]] = 0

    fat_ratio = fat_size / mask_size
    print(f'등심 영역중 지방의 비율: {fat_ratio}')

    return k, fat


def main():
    meat_g = cv2.imread('./meat.png', cv2.IMREAD_GRAYSCALE)
    meat = cv2.imread('./meat.png')
    mask = cv2.imread('./mask.png', cv2.IMREAD_GRAYSCALE)

    # otsu's method 적용
    k, fat = otsu_method_by_inter_class_variance(meat_g, mask)
    print(f'threshold: {k}')

    fat_3ch = np.zeros((fat.shape[0], fat.shape[1], 3), dtype=np.uint8)
    fat_3ch[:, :, 1] = fat

    # 원본 이미지에 dst 적용하기
    final = cv2.addWeighted(meat, 1, fat_3ch, 0.5, 0)

    cv2.imshow('meat', meat)
    cv2.imshow('fat_area', fat)
    cv2.imshow('fat_area_3ch', fat_3ch)
    cv2.imshow('final', final)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 보고서 첨부용
    cv2.imwrite('fat_area.png', fat)
    cv2.imwrite('final.png', final)

    return


if __name__ == '__main__':
    main()