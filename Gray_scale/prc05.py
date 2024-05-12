import cv2

src = cv2.imread('data/Lena.png')
rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

print(f'type(src) : {type(src)}')
print(f'src.dtype : {src.dtype}')
print(f'src.shape : {src.shape}')

cv2.imshow('Original', src)
cv2.imshow('RGB', rgb)
cv2.imshow('GRAY', gray)

cv2.waitKey()
cv2.destroyAllWindows()
