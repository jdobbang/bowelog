#일단은 혈변 부위 검출만 수행
import cv2

src = cv2.imread('blood1.jpg', cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

lower_stool = (0,80,0)
upper_stool = (120,255,120)
stool_ = cv2.inRange(hsv, lower_stool, upper_stool)
whole = cv2.bitwise_and(hsv, hsv, mask = stool_)
whole = cv2.cvtColor(whole, cv2.COLOR_HSV2BGR)
wholeCount = cv2.cvtColor(whole, cv2.COLOR_BGR2GRAY)

lower_red = (0,70,0)
upper_red = (6,255,255)
red_ = cv2.inRange(hsv, lower_red, upper_red)
blood = cv2.bitwise_and(hsv, hsv, mask = red_)
blood = cv2.cvtColor(blood, cv2.COLOR_HSV2BGR)
bloodCount = cv2.cvtColor(blood, cv2.COLOR_BGR2GRAY)

# get all non black Pixels
cntNotBlack1 = cv2.countNonZero(wholeCount)

# get all non black Pixels
cntNotBlack2 = cv2.countNonZero(bloodCount)

percentage = (cntNotBlack2 / cntNotBlack1)*100

print(percentage)
cv2.imshow("Original",src)
cv2.imshow("Blood", blood)
cv2.imshow("Whole", whole)
cv2.waitKey()
cv2.destroyAllWindows()