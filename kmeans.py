import cv2
import numpy as np
import tool
def kmeans(img,K=6,mask=None):
    img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imwrite('img/~test.png',mask*img)
    Z = img.reshape((-1,1))
    Z = np.float32(Z)
    #尝试使用opencv和sklearn的kmeans，感觉opencv更快一点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,_label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # label = KMeans(n_clusters=K, random_state=0).fit(Z).labels_
    _label = np.uint8(_label)
    count = np.bincount(_label.flatten())
    #目前实现思路是选第二大的分类label（第一大的是背景）
    a = np.sort(count)[-2]
    max_label = list(count).index(a)
    label = np.where(_label == max_label, 1, 0)
    label = np.uint8(label)
    res1 = label.reshape(img.shape)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30, 30))
    # res2 = cv2.morphologyEx(res1, cv2.MORPH_OPEN, kernel)
    # res3 = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel2)
    res2 = tool._open(res1)
    res3 = tool._close(res2)
    return res3 , _label.reshape(img.shape)
