import cv2
import numpy as np

src1 = np.zeros((200, 200))
src2 = np.zeros((200, 200), dtype=np.uint8)

src1[:, 100:200] = 1.
src2[:, 100:200] = 255

cv2.imshow('src1', src1)
cv2.imshow('src2', src2)

print(src1[100, 95:105])
print(src2[100, 95:105])

cv2.waitKey()
cv2.destroyAllWindows()
