import cv2
import numpy as np
import math

src = cv2.imread('blood1.jpg', cv2.IMREAD_COLOR)
src2 = cv2.imread('sample2.jpg', cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(src2, cv2.COLOR_BGR2HSV)

lower_stool = (0,80,0)
upper_stool = (120,255,120)

stool_ = cv2.inRange(hsv, lower_stool, upper_stool)
whole = cv2.bitwise_and(hsv, hsv, mask = stool_)
whole = cv2.cvtColor(whole, cv2.COLOR_HSV2RGB)

stool_2 = cv2.inRange(hsv2, lower_stool, upper_stool)
whole2 = cv2.bitwise_and(hsv2, hsv2, mask = stool_2)
whole2 = cv2.cvtColor(whole2, cv2.COLOR_HSV2RGB)

#thresh
wholeContour = cv2.cvtColor(whole, cv2.COLOR_RGB2GRAY)
ret,thresh = cv2.threshold(wholeContour, 10, 255, cv2.THRESH_BINARY)

#thresh
wholeContour2 = cv2.cvtColor(whole2, cv2.COLOR_RGB2GRAY)
ret2,thresh2 = cv2.threshold(wholeContour2, 10, 255, cv2.THRESH_BINARY)

#mophology
kernel = np.ones((20, 20), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("thresh",thresh)

#mophology
thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
cv2.imshow("thresh2",thresh2)

canny1 = cv2.Canny(thresh,50,150)
cv2.imshow("canny1",canny1)
canny2 = cv2.Canny(thresh2,50,150)
cv2.imshow("canny2",canny2)

#contour
contours, hierarchy = cv2.findContours(canny1, 2,1)
contours2, hierarchy2 = cv2.findContours(canny2, 2,1)

cnt1 = contours[0]
cnt2 = contours2[0]

similarity = cv2.matchShapes(cnt1,cnt2,1,0)
print("경계면 유사도",similarity)

for i in range(len(contours)):
    cv2.drawContours(src, [contours[i]], 0, (0, 0, 255), 2)  
    cv2.imshow("src", src)

for i in range(len(contours2)):
    cv2.drawContours(src2, [contours2[i]], 0, (0, 0, 255), 2)
    cv2.imshow("src2", src2)
    
cv2.waitKey(0)
cv2.destroyAllWindows()