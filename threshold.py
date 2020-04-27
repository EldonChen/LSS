import cv2
import numpy as np
import tool
img = None
mask = None


def threshold(img,low,high):
    r = 15
    ret1, image_bin_L = cv2.threshold(img, low, 255,cv2.THRESH_BINARY)
    ret2, image_bin_H = cv2.threshold(img, high, 255,cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(image_bin_L, image_bin_H)
    mask = _open(mask,r)
    mask = _close(mask,r)
    return mask

def _open(img,r=10,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(shape,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def _close(img,r=10,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(shape,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def mythreshold(x):
    global mask
    low = cv2.getTrackbarPos("threshold_L", "image")
    high = cv2.getTrackbarPos("threshold_H", "image")
    mask = threshold(img,low,high)
    cv2.imshow("mask",mask)
    _mask = tool.get_biggest_obj(mask)
    cv2.imshow("image",cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=_mask))
    cv2.imshow("_mask",_mask)


def main(path = './img/simple.png'):   
    global img
    global mask
    img = cv2.imread(path,-1)
    mask = np.ones_like(img)*255
    cv2.namedWindow("image")
    cv2.namedWindow("mask")
    cv2.namedWindow("_mask")
    cv2.createTrackbar("threshold_L", "image", 0, np.max(img), mythreshold) 
    cv2.createTrackbar("threshold_H", "image", np.max(img), np.max(img), mythreshold) 

    cv2.imshow("mask",mask)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite('img/mask.png',mask)
    mask = tool.get_biggest_obj(mask)
    return mask

if __name__ =='__main__':
    main()