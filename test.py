import cv2
import numpy as np

path='./img/simple.png'

img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
print(img)
Z = img.reshape((-1,1))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 6
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
print(label)
count = np.bincount(label.flatten())
a = np.sort(count)[-2]
max_label = list(count).index(a)
label = np.where(label == max_label, 1, 0)
label = np.uint8(label)
res1 = label.reshape(img.shape)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15, 15))
res2 = cv2.morphologyEx(res1, cv2.MORPH_OPEN, kernel)
res3 = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel2)

print(img.shape)
print(res2.shape)

# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
# count = np.bincount(label)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imshow('res1',img*res1)
cv2.waitKey(0)
cv2.imshow('res2',img*res2)
cv2.waitKey(0)
cv2.imshow('res3',img*res3)
cv2.waitKey(0)
# cv2.destroyAllWindows()