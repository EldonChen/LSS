import cv2
import numpy as np
import math
# from scipy.optimize import leastsq
from sklearn import mixture
import matplotlib.mlab
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

import threshold
import kmeans
import tool

import scipy.stats as sta

# def _open(img,r=5,shape=cv2.MORPH_ELLIPSE):
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r, r))
#     return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# def _close(img,r=30,shape=cv2.MORPH_ELLIPSE):
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r, r))
#     return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# def remove_fine_obj(img,area=30):
#     _, contours, hierarchy	= cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     print(hierarchy)
#     for i in range(hierarchy.shape[0]):
#         if cv2.contourArea(contours[i]) <= area:
#             contours.delete()

def func(x, a,u, sig):
    return  a*(np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig))

def residuals(p, x, y):
    regularization = 0.1  # 正则化系数lambda
    ret = y - func(x, p[0],p[1],p[2])
    return ret

path='./img/046.png'
# img = cv2.imread(path,-1)
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

# rmask = np.zeros(img.shape[:2], np.uint8)
# rmask[120:350, 70:350] = 255

# rmask = None

rmask = threshold.main(path)
rmask = tool._dilate(rmask,r=30)

# rmask = cv2.imread('img/mask.png',cv2.IMREAD_GRAYSCALE)

K = 6
mask,label = kmeans.kmeans(img,K=K,mask = rmask)

# tool.remove_fine_obj(mask)
ret = cv2.bitwise_and(img, img, mask=mask)

Imax = int(img.max())
Imin = int(img.min())
IBin = int(Imax - Imin +1)
Ibox = [Imin,Imax]

img_mask = np.ma.masked_equal(cv2.bitwise_and(img, img, mask=mask),0)

hist = cv2.calcHist([cv2.bitwise_and(img, img, mask=mask)],[0],mask,[IBin],Ibox)
hist = np.convolve(hist.flatten(),[1/3,1/3,1/3],'same')
hist = hist/np.sum(mask)


sigma_Prec = np.std(img_mask.flatten()) 
mu_Prec = np.mean(img_mask.flatten()) 

y_Prec = sta.norm.pdf(range(IBin), mu_Prec, sigma_Prec)


x = range(IBin)
# x = np.array(x).reshape(-1, 1)
# y_ = hist.flatten().reshape(-1, 1)
x = np.array(x)
y_ = hist.flatten()
# 尝试用最小二乘拟合获得强度直方图（可惜失败了）
# pars = np.random.rand(3)
# r = leastsq(residuals, pars, args=(x, y_))
# popt = r[0]
# print(r)
# y = [func(i, popt[0],popt[1],popt[2]) for i in range(256)]

#另一种尝试拟合（效果很一言难尽）
dataset = np.column_stack((x,y_))

# clf = mixture.GaussianMixture(1)
# clf.fit(x,y=y_)
# m1= clf.means_
# w1= clf.weights_
# c1= clf.covariances_
# def fun(x):
#     return w1*matplotlib.mlab.normpdf(x,m1,np.sqrt(c1[0]))[0]

def gaussian(x,*param):
    return param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))+\
           param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))

popt,pcov = curve_fit(gaussian,x,y_,p0=[3,4,3,6,1,1])

# y = [fun(i) for i in range(IBin)]
y = gaussian(x,*popt)
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
# plt.subplot(224),plt.plot(hist,'c'),plt.plot(range(IBin),y,'r')
plt.subplot(224),plt.plot(hist,'c'),plt.plot(range(IBin),y_Prec,'r')
# plt.subplot(235),plt.plot(range(IBin),y_Prec,'r')

plt.show()
