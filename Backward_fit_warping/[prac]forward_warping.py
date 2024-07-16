import cv2
import numpy as np


def forward(src, M, m_type=''):
    h, w = src.shape
    src = src.astype(np.float32)

    # 결과 영상의 크기를 입력 영상의 크기와 동일하게 설정
    dst = np.zeros((h, w), dtype=np.float32)

    # 변환 타입이 scaling일 경우 결과 영상의 크기를 조정
    if m_type == 'scaling':
        ratio_h = M[0, 0]
        ratio_w = M[1, 1]
        dst = np.zeros((int(ratio_h * h), int(ratio_w * w)), dtype=np.float32)

    h_, w_ = dst.shape
    N = np.zeros(dst.shape)

    for row in range(h):
        for col in range(w):
            # P: Point vector (x, y, 1)
            P = np.array([
                [col],  # x
                [row],  # y
                [1]
            ])

            P_dst = np.dot(M, P)  # (x,y,1) vector와 변환 행렬을 행렬 곱함
            dst_col = P_dst[0][0]  # x
            dst_row = P_dst[1][0]  # y

            # ceil은 수학기호로 [x]를 의미 즉 i >= x인 최소 정수를 의미.
            dst_col_right = int(np.ceil(dst_col))
            dst_col_left = int(dst_col)

            dst_row_bottom = int(np.ceil(dst_row))
            dst_row_top = int(dst_row)

            # index를 초과하는 부분은 값을 채우지 않음.

            if dst_row_top < 0 or dst_col_left < 0 or dst_row_bottom >= h_ or dst_col_right >= w_:
                continue

            # dst_row,col이 정수이면 original 좌표에서 값을 가져온 후 counting을 한다.
            dst[dst_row_top, dst_col_left] += src[row, col]
            N[dst_row_top, dst_col_left] += 1

            # dst_col 즉 x' 좌표가 소수라면
            if dst_col_right != dst_col_left:
                dst[dst_row_top, dst_col_right] += src[row, col]
                N[dst_row_top, dst_col_right] += 1

            # dst_row 즉 y' 좌표가 소수라면
            if dst_row_bottom != dst_row_top:
                dst[dst_row_bottom, dst_col_left] += src[row, col]
                N[dst_row_bottom, dst_col_left] += 1

            # dst_col, dst_row 즉 x',y' 모두 좌표가 소수라면
            if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                dst[dst_row_bottom, dst_col_right] += src[row, col]
                N[dst_row_bottom, dst_col_right] += 1

    N[N == 0] = 1
    dst = np.round(dst / N)
    dst = dst.astype(np.uint8)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    h, w = src.shape

    # M_ro = np.array([[np.cos(np.deg2rad(30)), -np.sin(np.deg2rad(30)), 0],
    #                  [np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30)), 0],
    #                  [0, 0, 1]])

    # M_sh = ???

    # M_sc = np.array([[4, 0, 0],
    #                  [0, 4, 0],
    #                  [0, 0, 1]])

    M_ro = np.array([[np.cos(np.deg2rad(20)), -np.sin(np.deg2rad(20)), 0],
                     [np.sin(np.deg2rad(20)), np.cos(np.deg2rad(20)), 0],
                     [0, 0, 1]])

    M_sh = np.array([[1, 0.2, 0],
                     [0,   1, 0],
                     [0,   0, 1]])

    M_sc = np.array([[1.2,   0, 0],
                     [  0, 1.2, 0],
                     [  0,   0, 1]])
    #M = M_ro @ M_sh @ M_sc
    M = M_sc @ M_sh @ M_ro

    print(M)
    # M_tr = ???

    #dst_for = forward(src, M_ro)
    # dst_for = forward(src, M_sh)
    #dst_for = forward(src, M_sc)
    dst_for = forward(src, M, 'scaling')
    # dst_for = forward(src, M_tr)

    cv2.imshow('dst_for', dst_for)
    cv2.imwrite('dst.png', dst_for)

    cv2.waitKey()
    cv2.destroyAllWindows()
