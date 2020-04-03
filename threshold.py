import cv2
import numpy as np
 
def nothing(x):
    pass
        
def mythreshold(image):
    #open image 
    image_org = cv2.imread(image)    
    #transe image to gray
    image_gray = cv2.cvtColor(image_org, cv2.COLOR_RGB2GRAY) 

    cv2.namedWindow("image")
    cv2.createTrackbar("threshold_L", "image", 0, 255, nothing) 
    cv2.createTrackbar("threshold_H", "image", 255, 255, nothing) 

    while True:
        mythreshold_L = cv2.getTrackbarPos("threshold_L", "image")
        mythreshold_H = cv2.getTrackbarPos("threshold_H", "image")  
        ret, image_bin_L = cv2.threshold(image_gray, mythreshold_L, 1,cv2.THRESH_BINARY)
        ret, image_bin_H = cv2.threshold(image_gray, mythreshold_H, 1,cv2.THRESH_BINARY_INV)
        image_bin = cv2.bitwise_and(image_bin_L, image_bin_H)
        cv2.imshow("image",image_bin)          
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    return image_bin
    
def main():
    path = './img/simple.png'
    mythreshold(path)   
            
if __name__ =='__main__':
    main()