import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
"""
def plot_colors(hist, cent):
    start = 0
    end = 0
    myRect = np.zeros((50, 300, 3), dtype="uint8")
    tmp = hist[0]
    tmpC = cent[0]
    for (percent, color) in zip(hist, cent):
        if(percent > tmp):
            tmp = percent
            tmpC = color
    end = start + (tmp * 300) # try to fit my rectangle 50*300 shape
    cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
                  tmpC.astype("uint8").tolist(), -1)
    start = end
    #rest will be black. Convert to black
    for (percent,color) in zip(hist, cent):
        end = start + (percent * 300)  # try to fit my rectangle 50*300 shape
        if(percent != tmp):
            color = [0, 0, 0]
            cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
                      color, -1) #draw in a rectangle
            start = end
    return myRect
"""
def stoolOnly(hsv,lower,upper):
        
    stoolClr = cv2.inRange(hsv, lower, upper)
    stoolShp = cv2.bitwise_and(hsv, hsv, mask = stoolClr)
    stoolShp = cv2.cvtColor(stoolShp, cv2.COLOR_HSV2RGB)
    
    
    stoolgray = cv2.cvtColor(stoolShp, cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(stoolgray, 10, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((11, 11), np.uint8)
    result2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    canny = cv2.Canny(result2,50,150)
    
    cv2.imshow("Stool Only", stoolShp)
    cv2.imshow("Stool Binary", thresh)   
    cv2.imshow("Close",result2) 
    cv2.imshow("Canny",canny)
    cv2.waitKey()
    cv2.destroyAllWindows()

def plot_colors2(hsv,hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        
        #각 color의 hsv 값과 percentage 추출
        color3d =  np.reshape(color, ( 1, 1, 3))
        color_hsv = cv2.cvtColor(color3d.astype("uint8"), cv2.COLOR_BGR2HSV)
        print(color_hsv)
        print(percent*100)
        
        # 하얀색(변기 or 회색 Stool)일 경우 s<40으로 일단 제외
        if color_hsv[0,0,1] <40:
            print("하얀색 변기 or 회색 stool 제외")
            continue
        
        # 갈색 Stool, 이런식으로 hsv 범위 설정하여 변의 색깔
        if (color_hsv[0,0,0] in range (0,120) and color_hsv[0,0,1] in range (80,255) and color_hsv[0,0,2] in range(0,101)):
            print("Stool은 고동색입니다!!!")
            lower = (0,80,0)
            upper = (120,255,101)            
          
        
        startX = endX
        
    stoolOnly(hsv, lower, upper)
    # return the bar chart
    return bar

img = cv2.imread("pic/stool3.jpg")


cv2.imshow("original",img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
clt = KMeans(n_clusters=3) #cluster number
clt.fit(img)

hist = find_histogram(clt)
bar = plot_colors2(hsv, hist, clt.cluster_centers_)

plt.axis("off")
plt.imshow(bar)
plt.show()