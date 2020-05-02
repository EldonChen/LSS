import cv2

def _open(img,r=10,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(shape,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations=1)

def _close(img,r=10,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(shape,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=1)

def _erosion(img,r=10,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(shape,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel,iterations=1)    

def _dilate(img,r=10,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(shape,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel,iterations=1)    

def remove_fine_obj(img,area=30):
    _, contours, hierarchy	= cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(hierarchy)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) <= area:
            cv2.drawContours(img,[contours[i]],0,0,-1)
    return img

def get_biggest_obj(img):
    _, contours, hierarchy	= cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    max = 0
    for i in range(len(contours)):
        if max < cv2.contourArea(contours[i]):
            max = cv2.contourArea(contours[i])
    return remove_fine_obj(img,max-1)