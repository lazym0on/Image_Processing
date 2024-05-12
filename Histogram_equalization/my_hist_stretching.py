import numpy as np
import matplotlib.pyplot as plt
import cv2


def my_cal_Hist(img):
    """
    Argument info
    img: gray scale image (H x W)

    variable info
    hist : 이미지의 픽셀 값들의 빈도수를 세는 1차원 배열
    hist의 index는 이미지의 픽셀 값을 의미 i.e) pixel 값 18 == index 18

    return info
    hist: 입력 이미지의 각 픽셀 빈도수를 나타내는 배열
    """
    h, w = img.shape
    # 주어진 이미지의 가진 1차원 배열 생성
    # 256 : 0 ~ 255 범위의 픽셀
    hist = np.zeros(256, dtype=np.int32)

    # histogram bin
    for row in range(h):
        for col in range(w):
            intensity = img[row, col]
            hist[intensity] += 1

    return hist

def plot_histogram(src):

    # histogram 계산
    hist = my_cal_Hist(src)
    bin_x = np.arange(len(hist))
    plt.bar(bin_x, hist, width=0.8, color='g')
    plt.title('my_histogram')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel frequency')
    plt.show()


def main():

    src = cv2.imread('./fruits.jpg', cv2.IMREAD_GRAYSCALE)

    low_contrast_img = np.round(src / 3).astype(np.uint8)

    # low contrast 이미지 확인
    cv2.imshow('low contrast image', low_contrast_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # low contrast 이미지 histogram 확인
    plot_histogram(low_contrast_img)

    cur_max = low_contrast_img.max()
    cur_min = low_contrast_img.min()

    # 직접 설정 - hyper parameter
    # 잘 모르겠으면 max 255, min 0으로 해볼 것
    target_max = 255
    target_min = 0


    ################################################
    # TODO
    # linear transformation function 완성하기
    # gradient : 기울기

    low_contrast_img = low_contrast_img.astype(np.float32)
    gradient = (target_max - target_min) / (cur_max - cur_min)
    high_contrast_img = gradient * (low_contrast_img - cur_min) + target_min
    ##################################################
    high_contrast_img = np.round(high_contrast_img).astype(np.uint8)

    # high contrast 이미지  확인
    cv2.imshow('my histogram stretching', high_contrast_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # high contrast 이미지 저장
    cv2.imwrite('my_histogram_strecthing_imgs.png', high_contrast_img)

    # high contrast 이미지 histogram 확인
    plot_histogram(high_contrast_img)

    return

if __name__ == '__main__':
    main()