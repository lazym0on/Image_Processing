import cv2
import numpy as np

src1 = np.zeros((200, 200))
src2 = np.ones((200, 200)) / 2
src3 = np.zeros((200, 200), dtype=np.uint8)
src4 = np.ones((200, 200), dtype=np.uint8)
src5 = np.full((200, 200), 255, dtype=np.uint8)

cv2.imshow('src1', src1)
cv2.imshow('src2', src2)
cv2.imshow('src3', src3)
cv2.imshow('src4', src4)
cv2.imshow('src5', src5)

print('<float>')
print(f'src1.shape : {src1.shape}')
print(f'src2.shape : {src2.shape}')
print(f'src1[0, 0] : {src1[0, 0]}', f'src2[0, 0] : {src2[0, 0]}')
print('<uint8>')
print(f'src3.shape : {src3.shape}')
print(f'src4.shape : {src4.shape}')
print(f'src5.shape : {src5.shape}')
print(f'src3[0, 0] : {src3[0, 0]}, ', f'src4[0, 0] : {src4[0, 0]}, ',
      f'src5[0, 0] : {src5[0, 0]}')

cv2.waitKey()
cv2.destroyAllWindows()
