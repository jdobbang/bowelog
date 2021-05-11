# Import the necessary libraries
import cv2
import numpy as np
from test_sobel import sobel_edge_detection 
import matplotlib.pyplot as plt

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

# Read the image as a grayscale image
img = cv2.imread('stool3.jpg')
#cv2.imshow("original",img)
img = cv2.GaussianBlur(img, (7,7), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_stool = (0,80,0)
upper_stool = (120,255,120)
stool_ = cv2.inRange(hsv, lower_stool, upper_stool)
whole = cv2.bitwise_and(hsv, hsv, mask = stool_)
whole = cv2.cvtColor(whole, cv2.COLOR_HSV2BGR)
wholeCount = cv2.cvtColor(whole, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret,th = cv2.threshold(wholeCount, 10, 255, 0)
kernel2 = np.ones((11, 11), np.uint8)
morph1 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2)
morph2 = cv2.morphologyEx(morph1, cv2.MORPH_OPEN, kernel2)

#canny edge detail
edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gradient_magnitude, gradient_direction = sobel_edge_detection(morph2, edge_filter, convert_to_degree=True, verbose="verbose")


#canny edge
canny = cv2.Canny(morph2,50,200)

contours, hierarchy = cv2.findContours(canny, 2,1)

# 미분 변화 가장 급격한 방향의 직선을 구하여 skeleton과의 교점을 구해 그 두점사이의
# 거리를 계산한다.

points = 5
length = len(contours)
background = np.zeros(morph2.shape,np.uint8)
for i in range(length):
    
        #print("i는" , i)
        cv2.drawContours(background, [contours[i]], 0, (255, 255, 255), 2)
        # 경계선 포인트 정하
        for j in range(points):
            step = int( len(contours[i])/points ) * j
            cv2.circle(background, tuple(contours[i][step][0]) , 5 ,(255,255,255),-1)
           
            # 경계선 포인트 직선 그리기
            xd = contours[i][step][0][0] - contours[i][step+1][0][0]
            yd = contours[i][step][0][1]- contours[i][step+1][0][1]
            slope = -1*xd/yd
            x = (contours[i][step][0][0] + contours[i][step+1][0][0]) /2
            y = (contours[i][step][0][1] + contours[i][step+1][0][1]) /2
            
            intercept = y - slope * x
            abline(slope,intercept)
    
# Step 1: Create an empty skeleton
size = np.size(morph2)
skel = np.zeros(morph2.shape, np.uint8)

# Get a Cross Shaped Kernel
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    open = cv2.morphologyEx(morph2, cv2.MORPH_OPEN, kernel)
    #Step 3: Substract open from the original image
    temp = cv2.subtract(morph2, open)
    #Step 4: Erode the original image and refine the skeleton
    eroded = cv2.erode(morph2, kernel)
    skel = cv2.bitwise_or(skel,temp)
    morph2 = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(morph2)==0:
        break
    
#kernel3= np.ones((3, 3), np.uint8)
#skel = cv2.dilate(skel, kernel3)
#skel = cv2.erode(skel, kernel)

compare = background + skel
#cv2.imshow("contour",background)
#cv2.imshow("skel",skel)
#cv2.imshow("compare",compare)
plt.imshow(compare)
plt.show()
# Displaying the final skeleton
cv2.waitKey(0)
cv2.destroyAllWindows()