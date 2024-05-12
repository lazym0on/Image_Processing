from operator import index

import numpy as np
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

def my_calcHist(src):
    ###############################
    # TODO                        #
    # my_calcHist완성             #
    # src : input image           #
    # hist : src의 히스토그램      #
    ###############################

    h, w = src.shape
    # 주어진 이미지의 가진 1차원 배열 생성
    hist = np.zeros(256, dtype=np.uint32)

    # histogram bin
    for row in range(h):
        for col in range(w):
            intensity = src[row, col]
            hist[intensity] += 1

    print(f"hist : \n {hist}")
    return hist

def my_normalize_hist(hist, pixel_num):
    ########################################################
    # TODO                                                 #
    # my_normalize_hist완성                                #
    # hist : 히스토그램                                     #
    # pixel_num : image의 전체 픽셀 수                      #
    # normalized_hist : 히스토그램값을 총 픽셀수로 나눔      #
    ########################################################

    normalized_hist = hist / pixel_num
    print(f"normalized_hist : \n {normalized_hist}")
    return normalized_hist


def my_PDF2CDF(pdf):
    ########################################################
    # TODO                                                 #
    # my_PDF2CDF완성                                       #
    # pdf : normalized_hist                                #
    # cdf : pdf의 누적                                     #
    ########################################################

    cdf = np.zeros(pdf.shape)
    cdf[0] = pdf[0]
    for i in range(1, len(pdf)):
        cdf[i] = cdf[i-1] + pdf[i]
    print(f"cdf : \n {cdf}")
    return cdf

def my_denormalize(normalized, gray_level):
    ########################################################
    # TODO                                                 #
    # my_denormalize완성                                   #
    # normalized : 누적된pdf값(cdf)                        #
    # gray_level : max_gray_level                          #
    # denormalized : normalized와 gray_level을 곱함        #
    ########################################################

    denormalized = normalized * gray_level
    print(f"my_denormalize : \n {denormalized}")

    return denormalized


def my_calcHist_equalization(denormalized, hist):
    ###################################################################
    # TODO                                                            #
    # my_calcHist_equalization완성                                    #
    # denormalized : output gray_level(정수값으로 변경된 gray_level)   #
    # hist : 히스토그램                                                #
    # hist_equal : equalization된 히스토그램                           #
    ####################################################################
    hist_equal = np.zeros(hist.shape)
    prefix_sum = hist[0]

    print(f"output gray level floor : \n {denormalized}")
    for i in range(1, 255):
        if denormalized[i - 1] == denormalized[i]:
            prefix_sum += hist[i - 1]
        else:
            hist_equal[denormalized[i - 1]] = prefix_sum + hist[i - 1]
            prefix_sum = 0
    # i = 255일때 집계 되지 않은 값 처리
    hist_equal[denormalized[255]] += hist[255] + prefix_sum

    print(f"hist_equal : \n {hist_equal}")

    return hist_equal


def my_equal_img(src, output_gray_level):
    ###################################################################
    # TODO                                                            #
    # my_equal_img완성                                                #
    # src : input image                                               #
    # output_gray_level : denormalized(정수값으로 변경된 gray_level)   #
    # dst : equalization된 결과 이미지                                 #
    ####################################################################

    (h,w) = src.shape
    dst = np.zeros((h,w),dtype=np.uint8)

    print(f"src pixel : \n {src}")

    for row in range(h):
        for col in range(w):
            dst[row, col] = output_gray_level[src[row, col]]

    print(f"dst pixel : \n {dst}")
    return dst



def my_hist_equal(src, type='original'):

    """
    1. histogram을 구한다 (my_calHist)
    2. histogram을 총 픽셀수로 나눈다(my_normalize_hist)
    3. 누적 분포(CDF)를 구한다 (my_PDF2CDF)
    4. 누적 시킨 값에 gray level의 최댓값(255)을 곱한다. (my_denormalize)
    5. 반올림 또는 버림을 하여 정수값으로 변환한다.
    6. 구해진 정수값을 사용하여 histogram equailization 결과를 반환한다. (my_calcHist_equalization)
    """

    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    # show mapping function
    ###################################################################
    # TODO                                                            #
    # plt.plot(???,???)완성                                           #
    # plt.plot(x축, y축)
    # x축 : 0 ~ 255 범위의 정수
    # y축 : 0 ~ 255 각각의 정수 값에 해당하는 equalization 변환 픽셀 값
    ###################################################################

    plt.plot(np.arange(len(histogram)), output_gray_level)
    plt.title(type + ' mapping function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal

def plot_equal_histogram(dst, hist_equal,type='original'):

    plt.figure(figsize=(8, 5))
    binX = np.arange(len(hist_equal))
    plt.title(type +' histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

def plot_before_equal_histogram(src, original_hist,type='original'):
    plt.figure(figsize=(8, 5))
    binX = np.arange(len(original_hist))
    plt.title(type)
    plt.bar(binX, original_hist, width=0.5, color='g')
    plt.show()


if __name__ == '__main__':

    src = cv2.imread('./Lena.png', cv2.IMREAD_GRAYSCALE)

    ###################################################################
    # TODO
    # src_add, src_sub, src_div, src_mul 완성
    # cv2.add, cv2.subtract cv2.divide cv2.multiply 사용 시 감점
    # src_add : src + 64
    # src_sub : src - 64
    # src_div : src / 3
    # src_mul : src * 2
    ###################################################################

    src_add = cv2.add(src, 64)
    src_add = np.round(src_add).astype(np.uint8)

    src_sub = cv2.subtract(src, 64)
    src_sub = np.round(src_sub).astype(np.uint8)

    src_div = cv2.divide(src, 3)
    src_div = np.round(src_div).astype(np.uint8)

    src_mul = cv2.multiply(src, 2)
    src_mul = np.round(src_mul).astype(np.uint8)

    # original
    src_hist = my_calcHist(src)
    dst, hist_src_equal = my_hist_equal(src)

    # src_add
    src_add_hist = my_calcHist(src_add)
    dst_add, hist_add_equal = my_hist_equal(src_add,type='src_add')

    # src_sub
    src_sub_hist = my_calcHist(src_sub)
    dst_sub, hist_sub_equal = my_hist_equal(src_sub,type='src_sub')

    # src_div
    src_div_hist = my_calcHist(src_div)
    dst_div, hist_div_equal = my_hist_equal(src_div,type='src_div')

    # src_mul
    src_mul_hist = my_calcHist(src_mul)
    dst_mul, hist_mul_equal = my_hist_equal(src_mul, type='src_mul')

    cv2.imshow('original',src)
    cv2.imshow('original equal equal_result', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('original.png', src)
    cv2.imwrite('original_equal_result.png', dst)

    cv2.imshow('src_add', src_add)
    cv2.imshow('src_add equal equal_result', dst_add)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('src_add.png', src_add)
    cv2.imwrite('src_add_equal_result.png', dst_add)

    cv2.imshow('src_sub', src_sub)
    cv2.imshow('src_sub equal equal_result', dst_sub)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('src_sub.png', src_sub)
    cv2.imwrite('src_sub_equal_result.png', dst_sub)

    cv2.imshow('src_div', src_div)
    cv2.imshow('src_div equal equal_result', dst_div)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('src_div.png', src_div)
    cv2.imwrite('src_div_equal_result.png', dst_div)

    cv2.imshow('src_mul', src_mul)
    cv2.imshow('src_mul equal equal_result', dst_mul)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('src_mul.png', src_mul)
    cv2.imwrite('src_mul_equal_result.png', dst_mul)

    plot_before_equal_histogram(src, src_hist, type='original')
    plot_equal_histogram(dst, hist_src_equal, type='original')

    plot_before_equal_histogram(src_add, src_add_hist, type='src_add')
    plot_equal_histogram(dst_add, hist_add_equal, type='src_add')

    plot_before_equal_histogram(src_sub, src_sub_hist, type='src_sub')
    plot_equal_histogram(dst_sub, hist_sub_equal, type='src_sub')

    plot_before_equal_histogram(src_div, src_div_hist, type='src_div')
    plot_equal_histogram(dst_div, hist_div_equal, type='src_div')

    plot_before_equal_histogram(src_mul, src_mul_hist, type='src_mul')
    plot_equal_histogram(dst_mul, hist_mul_equal, type='src_mul')

