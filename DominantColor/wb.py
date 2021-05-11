import cv2 as cv
# import numpy as np
 
def whiteBalance(img):
    
    #split
    r, g, b = cv.split(img)
    r_avg = cv.mean(r)[0]
    g_avg = cv.mean(g)[0]
    b_avg = cv.mean(b)[0]
     
     # Find the gain of each channel
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
     
    #balance
    r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
     
    balance_img = cv.merge([r, g, b])
    
    return balance_img