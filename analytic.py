import cv2
import numpy as np
import imutils

#function to find center of a line
def center(line):
    x1,y1=line[0][0]
    x2,y2=line[1][0]
    
    x_c=int((x1+x2)/2)
    y_c=int((y1+y2)/2)
    
    return x_c,y_c

def detect_level(orig,dimensions,scale,debug=False):
    #copy image to a variable
    src=orig
    
    #resize: change scale if resizing is required
    scale=scale
    r_H=int(src.shape[0]*scale)
    r_W=int(src.shape[1]*scale)
    dim=(r_W,r_H)
    src=cv2.resize(src,dim)
    
    #load bottle dimensions
    x,y,w,h=dimensions
    
    #crop image to dimension
    src=src[y:y+h,x:x+w]
    
    src=src[int(0.1*h):int(h-(0.15*h))]
    src=src[:,int(0.05*w):int(w-(0.05*w))]
    
    #apply filters for later edge detection
    src = cv2.GaussianBlur(src, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    #linear brightness stretching
    a=3.2
    m1=np.mean(gray)-a*np.std(gray)
    m2=np.mean(gray)+a*np.std(gray)
    gray=255*((gray-m1)/(m2-m1))
    gray=np.uint8(gray)
    
    #apply a Sobel filter in the y direction
    ddepth = cv2.CV_16S
    scale_sob = 1
    delta = 10
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale_sob, delta=delta, borderType=cv2.BORDER_DEFAULT)
    filtered = cv2.convertScaleAbs(grad_y)
    
    filtered=cv2.addWeighted(filtered,1.2,filtered,0,0)
    
    grad_y = cv2.Sobel(filtered, ddepth, 0, 1, ksize=3, scale=scale_sob, delta=delta, borderType=cv2.BORDER_DEFAULT)
    filtered = cv2.convertScaleAbs(grad_y)
    
    if debug==True:
        cv2.imshow("frame",filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    #apply a series of erosions and dilations and thresholds
    filtered=cv2.dilate(filtered, None, iterations=2)
    filtered=cv2.erode(filtered, None, iterations=4)
    filtered=cv2.dilate(filtered, None, iterations=2)
    
    if debug==True:
        cv2.imshow("frame",filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    ret,thresh = cv2.threshold(filtered,127,255,0)
    
    cv2.imshow("frame",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #find contours from filtered edges
    contours=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(contours is not None):
        contours= imutils.grab_contours(contours)
        thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    
        cntlist=[]
        slopelist=[]
        
        min_contoursize=0.0004*(h*w)
        
        for i in range(len(contours)):
            #determine approx polygon from the contours
            epsilon = 0.2*cv2.arcLength(contours[i],True)
            approx = cv2.approxPolyDP(contours[i],epsilon,True)
            
            #remove all small contours
            if(cv2.contourArea(contours[i])>min_contoursize):
                #accept poly only if it is a line
                if(approx.shape[0]==2):
                    x1,y1=approx[0][0]
                    x2,y2=approx[1][0]
                    
                    #calculate slope of detected line
                    slope=abs((y2-y1)/(x2-x1))
                    cntlist.append(approx)
                    slopelist.append(slope)
                    
        if(len(cntlist))>0:
            #find the minimum slope
            minind=np.array(slopelist).argmin()
            if (slopelist[minind]<0.5):
                approx=cntlist[minind]
                #calculate center of line
                x_c,y_c=center(approx)
                #calculate height for actual picture
                height_final=y_c+int(0.1*h)+y
                return height_final
    return -1
