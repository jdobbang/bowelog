from tkinter import*
from tkinter.filedialog import*
import cv2
import numpy as np
from sklearn.cluster import KMeans
import glob

# gray world white balancing
def whiteBalance(img):
    
    #rgb extraction
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
     
     # Find the gain of each channel
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
     
    #balance
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
     
    balance_img = cv2.merge([r, g, b])
    
    return balance_img

#mayo score
def mayo (hsv,stool):
    
    #blood hsv range1
    lower_red = (0,100,10)
    upper_red = (7,255,255)    
    red_ = cv2.inRange(hsv, lower_red, upper_red)
    
    #blood hsv range2
    lower_red2 = (173,100,10)
    upper_red2 = (180,255,255)
    red_2 = cv2.inRange(hsv,lower_red2,upper_red2)
    
    #final blood hsv range
    red_ = red_ + red_2
    
    #blood hsv extraciton
    blood = cv2.bitwise_and(hsv, hsv, mask = red_)
    blood = cv2.cvtColor(blood, cv2.COLOR_HSV2BGR)
    bloodCount = cv2.cvtColor(blood, cv2.COLOR_BGR2GRAY)
   
    #morphology close + open
    kernel2 = np.ones((3, 3), np.uint8)
    blood1 = cv2.morphologyEx(bloodCount, cv2.MORPH_CLOSE, kernel2)
    blood2 = cv2.morphologyEx(blood1, cv2.MORPH_OPEN, kernel2)
    cv2.imshow("bloodONly",blood2)
    
    # stool/blood area count
    stoolArea = cv2.countNonZero(stool)
    bloodArea = cv2.countNonZero(blood2)
    
    # scoring : 
    if stoolArea == 0:
        print("stool is not found!")
    else:
        percentage = (bloodArea / stoolArea )*100
    
        if percentage == 0:
            score = 0
        
        if percentage > 0 and percentage <10:
            score = 1
        
        if percentage >= 10 and percentage <20:
            score = 2
        
        if percentage >= 20:
            score = 3

        print("혈량 = " , bloodArea , " 나누기 ", stoolArea ," 은 ", percentage , "이고 mayo score은 ", score )
        mayoState = str("\n혈량 = " + str(bloodArea) +  " 나누기 " +  str(stoolArea) + " 은 " + str(percentage) + "이고 mayo score은 " + str(score) )
    return mayoState

#historgram calculation
def find_histogram(model):
    
    #label number
    numLabels = np.arange(0, len(np.unique(model.labels_)) + 1)  #np.unique(model.labels_) == klustering의 개수 k
    #histogram
    (hist, _) = np.histogram(model.labels_, bins=numLabels)
    hist = hist.astype("float")    
    hist /= hist.sum()

    return hist

#color processing
def color(img,hsv,seg,hist, centroids):
   
    #dominant color : 50 x 300 color bar
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    i = 1
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        
        #각 color의 hsv 값과 percentage 추출
        color3d =  np.reshape(color, ( 1, 1, 3))
        hsvC = cv2.cvtColor(color3d.astype("uint8"), cv2.COLOR_BGR2HSV)
        
        # exception setting : gray stool, 청크린, black stool
        if hsvC[0,0,1] < 55 and hsvC[0,0,2] > 30:
            #print("하얀색 변기 or 회색 stool 제외!!!")
            continue
        
        if (hsvC[0,0,0] in range (80,135)):
            #print("청크린 제외!!!") # 
            continue
        
        if ( hsvC[0,0,2] < 30 ):
            #print("Stool은 검은색입니다!!!")
            #따로 컨투어 처리 해주기!
            continue
        
        #dominant stool에 대해 처리할때 오류나는 경우 발생
        if i == 1:
            color_hsv = hsvC
            dominant = percent
        if i != 1 and dominant < percent:
            color_hsv = hsvC
            dominant = percent
            
        startX = endX
        i = i + 1 
        
    print("dominant color의 hsv >> ", color_hsv)
    
    # stool color hsv setting
    if (color_hsv[0,0,0] in range (12,24) and color_hsv[0,0,2] in range(0,71)):
        print("Stool은 고동색(Dark Brown)입니다!!!")
        colorState= str("Stool은 고동색(Dark Brown)입니다!!!")
        lower, upper = bound(color_hsv)

    elif (color_hsv[0,0,0] in range (12,24) and color_hsv[0,0,2] in range(70,255)):
        print("Stool은 밝은 갈색(Brown)입니다!!!")
        colorState= str("Stool은 밝은 갈색(Brown)입니다!!!")
        lower, upper = bound(color_hsv)
        
    elif (color_hsv[0,0,0] in range (33,70)):
        print("Stool은 초록색(Green)입니다!!!")
        colorState= str("Stool은 초록색(Green)입니다!!!")
        lower, upper = bound(color_hsv)
        
    elif (color_hsv[0,0,0] in range (24,33)):
        print("Stool은 노란색(Yellowish)입니다!!!")
        colorState= str("Stool은 노란(Yellowish)입니다!!!")
        lower, upper = bound(color_hsv)
        
    elif (color_hsv[0,0,0] in range (0,12)) :
        print("Stool은 적색(Red)입니다!!!")
        colorState= str("Stool은 적색(Red)입니다!!!")
        lower, upper = bound(color_hsv)
        
    elif (color_hsv[0,0,0] in range (230,240)) :
        print("Stool은 적색( Red)입니다!!!")
        colorState= str("Stool은 적(Red)입니다!!!")
        lower, upper = bound(color_hsv)
    
    stool,contourSample= stoolOnly(img,seg,lower,upper)
    mayoState=mayo(hsv,stool)
    
    return bar,contourSample,mayoState,colorState
# hsv의 범위에 따라 해당 영역 추출 과정 시각화 함수
def stoolOnly(img,hsv,lower,upper):
        
    #hsv 범위에 따라 segmented image의 stool 부분 추출(영역 불분명)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    stoolClr = cv2.inRange(hsv, lower, upper)
    stoolShp = cv2.bitwise_and(hsv, hsv , mask = stoolClr)
    stoolShp = cv2.cvtColor(stoolShp, cv2.COLOR_HSV2BGR)
    
    #binary 이미지로 영역 추출(영역 불분명)
    stoolgray = cv2.cvtColor(stoolShp, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(stoolgray, 20, 255, cv2.THRESH_BINARY)
    
    #morphology(close >> open) 적용(영역 분명)
    kernel = np.ones((11, 11), np.uint8)
    result2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result3 = cv2.morphologyEx(result2, cv2.MORPH_OPEN, kernel)
    
    #stool 부분만 추출
    stoolstool = cv2.bitwise_and(img , img , mask = result3)
    
    #cany edge >> contour 추출
    canny = cv2.Canny(result3,50,150)
    contours, hierarchy = cv2.findContours(canny, 2,1)
    
    #중간 결과 출력

    cv2.imshow("Stool Binary", thresh)   
    cv2.imshow("Close + Open", result3)
    cv2.imshow("Canny",canny)
    for i in range(len(contours)):
        cv2.drawContours(img, [contours[i]], 0, (0, 0, 255), 2)
        cv2.imshow("src", img) 
    cv2.imshow("stoolstool",stoolstool)
    
    return result3 , contours

#dominant stool color에 대한 임의의 범위 설정
def bound(color_hsv):
    
        #h,s,v 각 요소별로 분
        color_hsv_ = color_hsv.astype("uint8").tolist()
        h = color_hsv_[0][0][0]
        s = color_hsv_[0][0][1]
        v = color_hsv_[0][0][2]
        
        # h,s,v 각 분포 
        hd = 5
        sd = 40
        vd = 10
        h_ = h-hd
        s_ = s-sd
        v_ = v-vd
        
        #exception handling
        if h_ < 0:
            h_ = 0
        
        if s_ < 0:
            s_ = 0
            
        if v_ < 0:
            v_ =0    
            
        if h_ > 180:
            h_ = 180
        
        if s_ > 255:
            s_ = 255
            
        if v_ > 255:
            v_ = 255  
    
        lower = (h_,s_,v_)
        upper = (h+hd,s+sd,v+vd)

        return lower , upper

#segmentation
def segmentation(img , K):
    
    #convert shape and data type
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2

#K determination
def elbow(X):
    X = X.reshape((X.shape[0] * X.shape[1],3)) #represent as row*column,channel number
    sse = []
    standard = 200000000 # 0.5*(10^9)
               
    for i in range(1,7):
        km = KMeans(n_clusters=i,init='k-means++',random_state=0)
        km.fit(X)
        sse.append(km.inertia_) 
        
        if km.inertia_ < standard:
            print("K는", i+2)
            K = i
            break
    return K + 2 

#main
def main(file_record):
       
    #program start
    img_ = cv2.imread(file_record['image'],cv2.IMREAD_COLOR)
    img_ = cv2.resize(img_, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    
    #white balancing
    img = whiteBalance(img_)
   
    #convert to hsv space
    hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
    
    #find images clustering number K
    K = elbow(img)
    
    #k-means clustering segmentation
    seg = segmentation(img,K)
    seg_ = seg.reshape((seg.shape[0] * seg.shape[1],3)) #represent as row*column,channel number
    model = KMeans(n_clusters=K) #cluster definition
    model.fit(seg_)#adapt this model to original img
    
    #histogram calculation
    hist = find_histogram(model) 
    
    #output definition
    global mayoState,colorState
    
    #주요 분석 함수
    bar,contourSample,mayoState,colorState = color(img ,hsv, seg, hist, model.cluster_centers_)
   
    #print
    cv2.imshow("segmented",seg)
    cv2.imshow("whiteBalance",img)
    cv2.imshow("original",img_)
    cv2.imshow("bar",bar)
    
#분석 output
def result(colorState,mayoState):
    
    label_C.config(text = colorState)
    label_M.config(text = mayoState)
    
#open file 기능
def OpenFile(file_record):
    file_record['image'] =  askopenfilename(title = "Select file",filetypes = (("image files1","*.JPG"), ("image files2","*.gif")))
    print(file_record['image'])

root = Tk()
filename_record = {}
colorState = {}
mayoState = {}

#gui title
root.title("Stool Project")
root.geometry("500x400")
root.resizable(0,0)

#파일 선택 메
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Open", command=lambda: OpenFile(filename_record))
root.config(menu=menubar)

#분석 버튼
cutestool = PhotoImage(file = "cutestool.gif")
btn = Button(root,text = "분석", image = cutestool, command = lambda: main(filename_record))
btn.pack()

#결과 출력 버튼
btn2 = Button(root,text = "결과",command = lambda: result(colorState,mayoState))
btn2.pack()

label_C = Label(root,text = "color")
label_M = Label(root,text = "mayo score")

label_C.pack()
label_M.pack()
root.mainloop()

