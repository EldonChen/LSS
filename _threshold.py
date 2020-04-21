import cv2
import numpy as np
 
def nothing(x):
    pass
        
def _open(img,r=5,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def _close(img,r=10,shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r, r))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def mythreshold(image):
    #open image 
    image_org = cv2.imread(image)
    #transe image to gray
    image_gray = cv2.cvtColor(image_org, cv2.COLOR_RGB2GRAY) 

    cv2.namedWindow("image")
    cv2.createTrackbar("threshold_L", "image", 0, np.max(image_gray), nothing) 
    cv2.createTrackbar("threshold_H", "image", np.max(image_gray), np.max(image_gray), nothing) 

    while True:
        mythreshold_L = cv2.getTrackbarPos("threshold_L", "image")
        mythreshold_H = cv2.getTrackbarPos("threshold_H", "image")  
        ret1, image_bin_L = cv2.threshold(image_gray, mythreshold_L, 255,cv2.THRESH_BINARY)
        ret2, image_bin_H = cv2.threshold(image_gray, mythreshold_H, 255,cv2.THRESH_BINARY_INV)
        image_bin = cv2.bitwise_and(image_bin_L, image_bin_H)
        image_bin = _open(image_bin)
        image_bin = _close(image_bin)
        cv2.imshow("image",image_bin*image_gray)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    return image_bin
    
def main():
    path = './img/053.png'
    mythreshold(path)   
            
if __name__ =='__main__':
    main()