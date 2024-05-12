import cv2
import numpy as np

src = np.zeros((300, 300, 3), dtype=np.uint8)

# [height, width] 좌표 개념
# r-value는 B, G, R 채널에 들어갈 값

src[0, 0] = [1, 2, 3]
src[0, 1] = [4, 5, 6]
src[0, 2] = [7, 8, 9]

print(src.shape)
# 1 2 3
print(src[0, 0, 0], src[0, 0, 1], src[0, 0, 2])
# [1 2 3]
print(src[0, 0])
#[[1 2 3], [4 5 6], [7 8 9], [0 0 0] ... ]
print(src[0, 0:5])
#print(src)

cv2.imshow('src', src)

cv2.waitKey()
cv2.destroyAllWindows()
