import cv2
import numpy as np
import math
# from scipy.optimize import leastsq
from sklearn import mixture
import matplotlib.mlab
from sklearn.cluster import KMeans
def kmeans(img,K=6,mask=None):
    img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite('img/~test.png',mask*img)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30, 30))
    res2 = cv2.morphologyEx(res1, cv2.MORPH_OPEN, kernel)
    res3 = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel2)
    return res3 , _label.reshape(img.shape)

def _open(img,r=5,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def _close(img,r=30,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def remove_fine_obj(img,area=30):
    _, contours, hierarchy	= cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    print(hierarchy)
    for i in range(hierarchy.shape[0]):
        if cv2.contourArea(contours[i]) <= area:
            contours.delete()

def func(x, a,u, sig):
    return  a*(np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig))

def residuals(p, x, y):
    regularization = 0.1  # 正则化系数lambda
    ret = y - func(x, p[0],p[1],p[2])
    return ret

path='./img/simple.png'
img = cv2.imread(path,-1)
# img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
rmask = np.zeros(img.shape[:2], np.uint8)
rmask[120:350, 70:350] = 255
K = 6
mask,label = kmeans(img,K=K,mask = rmask)

remove_fine_obj(mask)
ret = cv2.bitwise_and(img, img, mask=mask)

Imax = int(img.max())
Imin = int(img.min())
IBin = int(Imax - Imin +1)
Ibox = [Imin,Imax]

hist = cv2.calcHist([img],[0],mask,[IBin],Ibox)
hist = hist/np.sum(hist)
x = range(IBin)
x = np.array(x).reshape(-1, 1)
y_ = hist.flatten().reshape(-1, 1)

# 尝试用最小二乘拟合获得强度直方图（可惜失败了）
# pars = np.random.rand(3)
# r = leastsq(residuals, pars, args=(x, y_))
# popt = r[0]
# print(r)
# y = [func(i, popt[0],popt[1],popt[2]) for i in range(256)]

#另一种尝试拟合（效果很一言难尽）
dataset = np.column_stack((x,y_))

clf = mixture.GaussianMixture(1)
clf.fit(x,y=y_)
m1= clf.means_
w1= clf.weights_
c1= clf.covariances_
def fun(x):
    return w1*matplotlib.mlab.normpdf(x,m1,np.sqrt(c1[0]))[0]

y = [fun(i) for i in range(IBin)]
# hist = cv2.calcHist(images,channels,mask,histSize,ranges [,hist [,accumulate]])
'''参数说明：
images：uint8或float32类型的原图像。用方括号表示，即“[img]”;
channels：计算直方图的通道索引，也在方括号中给出.例如，如果输入是灰度图像，则其值为[0].对于彩色图像，可以通过[0]，[1]或[2]分别计算蓝色，绿色或红色通道的直方图.
mask：蒙版图像.要查找完整图像的直方图，它将显示为“无”.但是，如果要查找图像特定区域的直方图，则必须为其创建蒙版图像并将其作为蒙版.
histSize：这代表我们的BIN计数.需要在方括号中给出.对于满量程，我们通过[256].
ranges：这是我们的范围。通常，它是[0,256].
'''
from matplotlib import pyplot as plt
# plt.hist(hist,1,[0,256])
plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(label,'gray')
plt.subplot(223),plt.imshow(ret,'gray')
# lahist = cv2.calcHist([label],[0],None,[K],[0,K])
# plt.subplot(223),plt.plot(lahist,'c')
plt.subplot(224),plt.plot(hist,'c'),plt.plot(range(IBin),y,'r')
plt.show()
