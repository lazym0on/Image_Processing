import cv2
import numpy as np
import math
import time
import datetime

from tqdm import tqdm

def calc_bgr2gray(src):
    """
    :param src: color video
    :return: gray video
    """

    h, w, c = src.shape
    dst = np.zeros((h, w))

    for row in tqdm(range(h)):
        for col in range(w):
            ##############################################
            # TODO
            # B * 0.0721 + G * 0.7154 + R * 0.2125 공식을 사용
            # calc_bgr2gray 함수 완성
            # 반드시 2중 포문 형태를 유지
            # 라이브러리 사용할 필요 없음
            ##############################################

            dst[row, col] = (src[row, col, 0] * 0.0721 +
                             src[row, col, 1] * 0.7154 +
                             src[row, col, 2] * 0.2125)

    # dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    dst = (dst+0.5).astype(np.uint8)
    return dst


sec = 0.

video_file = './data/jin.avi'  # 동영상 경로
cap = cv2.VideoCapture(video_file) # 동영상 캡처 객체 생성

if cap.isOpened(): # 객체 초기화 정상
    fps = 30.0  # FPS, 초당 프레임 수
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 인코딩 포맷 문자
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 영상 저장을 위핸 객체 생성
    out = cv2.VideoWriter('./record_gray.avi', fourcc, fps, (w, h), isColor=False)

    while True:
        ret, frame = cap.read() # 다음 프레임 읽기
        if ret: # 프레임 읽기 정상
            start = time.time()
            frame_gray = calc_bgr2gray(frame)
            end = time.time()
            sec += (end - start)
            cv2.namedWindow('Gray', flags=cv2.WINDOW_NORMAL)
            cv2.namedWindow('Color', flags=cv2.WINDOW_NORMAL)
            cv2.imshow('Gray', frame_gray)
            cv2.imshow('Color', frame)
            out.write(frame_gray)
            if cv2.waitKey(int(1000/fps)) != -1: # 지연시간 = 1000/fps
                break
        else: # 프레임 읽기 비정상
            break
    out.release() # 저장 객체 소멸
else: # 객체 초기화 비정상
    print("can't open video")

result = datetime.timedelta(seconds=sec)
print(result)

cap.release() # 캡쳐 객체 소멸
cv2.destroyAllWindows()

