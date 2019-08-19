"""
This script detects width and water level of a bottle via webcam.
Detects bottle using YOLO and detects the water level using edge detection given than the level is between 80% of the bottle.

Run the script and align the bottle between the blue lines.
Make sure that the water level is as level with the camera as possible.
Once aligned, press d.
Enter the height of the bottle.
If no bottle was detected, or the water level was not detected, try again by varying the light and the angle.
If detected press q to exit the imageview.
"""

import cv2
import numpy as np
import analytic as water_lvl

print("WATER LEVEL AND BOTTLE WIDTH DETECTOR")
print("Press Ctrl+C to stop script.")

#define the YOLO detector
weights="yolo/yolov3_cust.weights"
config="yolo/yolov3_cust.cfg"
detector=cv2.dnn.readNetFromDarknet(config, weights)

while True:
    choice=int(input("1 - WebCam\n2 - Static Image\n::"));
    
    if choice==1:
        #start capturing the video stream from the webcam
        print("Align the bottle between the blue lines and press 'd'.")
        cap=cv2.VideoCapture(0)
        
        while True:
            _,frame=cap.read()
            #save the unaltered frame for later use
            disp=frame
            
            #extract height and width and draw the guide lines
            H=frame.shape[0]
            W=frame.shape[1]
            ph=int((5/100)*H)
            pw=int((50/100)*H)
            cv2.line(frame,(pw,ph),(W-pw,ph),(255,0,0),3)
            cv2.line(frame,(pw,H-ph),(W-pw,H-ph),(255,0,0),3)
            cv2.imshow("frame",frame)
            if cv2.waitKey(5) & 0xFF==ord('d'):
                break
        cv2.destroyAllWindows()
        cap.release()
    
    elif choice==2:
        #ask for file path and load file
        imgpath=input("Enter Full Image Path: ")
        disp=cv2.imread(imgpath)
    
    else:
        disp=None
        
    image=disp
    
    if image is not None:
        #define detector output layers
        ln=detector.getLayerNames()
        ln=[ln[i[0] - 1] for i in detector.getUnconnectedOutLayers()]
    
        #resize: change scale if you want to
        scale=1
        r_H=int(image.shape[0]*scale)
        r_W=int(image.shape[1]*scale)
        dim=(r_W,r_H)
        image=cv2.resize(image,dim)
    
        #redefine height and width according to resized image
        H=image.shape[0]
        W=image.shape[1]
        ph=int((5/100)*H)
        pw=int((50/100)*H)
    
        #construct blob from input (blob is basically normalization)
        blob=cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),swapRB=True, crop=False)
    
        #set the detector input as the blob ans forward the blob through the network
        detector.setInput(blob)
        outputs=detector.forward(ln)
    
        bounding_box=[]
        item_list=[]
        confidences=[]
    
        #filter the detections
        for output in outputs:
            for detection in output:
                classId=detection[5:].argmax()
                accuracy=detection[5:][classId]
                
                #check if bottle was detected with high confidence
                if(accuracy>0.4 and classId==39):
                    bbox_uns=detection[0:4]
                    bbox=bbox_uns*np.array([W,H,W,H])
                    
                    center_X,center_Y,width,height=bbox.astype(int)
                    
                    x=int(center_X-(width/2))
                    y=int(center_Y-(height/2))
                    
                    bounding_box.append([x,y,int(width),int(height)])
                    item_list.append(classId)
                    confidences.append(float(accuracy))
                    
    
        #get the NMS box for the bottle detections            
        idxs = cv2.dnn.NMSBoxes(bounding_box,confidences,0.5,0.1)
    
        #check if any bottles were detected. If yes proceed
        if(len(idxs)>0):
            #get the bounding box for the image
            bb=bounding_box[idxs[0][0]]
            
            #get the top corner index and height and width
            x,y,w,h=bb
            
            #determine the water height from the analytic.py script  
            height_water=water_lvl.detect_level(disp,(x,y,w,h),scale)
            
            if(height_water>0):
                height=int(input("Enter Height of bottle: "))
                
                #calculate bottle width from bottle height
                width=(w/h)*height
                
                #draw width line and text
                cv2.line(image,(x,y+h),(x+w,y+h),(0,0,255),3)
                cv2.putText(image,"{0:.2f}".format(width),(int(x+w/2)-10,int(y+h)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),5,cv2.LINE_AA,)
                cv2.putText(image,"{0:.2f}".format(width),(int(x+w/2)-10,int(y+h)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                
                #draw height line and text
                cv2.line(image,(x+w,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(image,"{0:.2f}".format(height),(int(x+w)+10,int(y+h/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),5,cv2.LINE_AA)
                cv2.putText(image,"{0:.2f}".format(height),(int(x+w)+10,int(y+h/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                
                #check if water line was detected script will return -1 if not detected
                if(height_water>0):
                    #calculate the height of the water line from the base of the bottle
                    wat_h= ((y+h)-height_water)
                    
                    #calculate water height from bottle height
                    act_h=(wat_h/h)*height
                    
                    #draw water line horizontal and vertical line and text
                    cv2.line(image,(pw,height_water),(W-pw,height_water), (0,255,0),3)
                    
                    cv2.line(image,(x,height_water),(x,y+h),(0,0,255),3)
                    cv2.putText(image,"{0:.2f}".format(act_h),(x+10,int((y+h+height_water)/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),5,cv2.LINE_AA)
                    cv2.putText(image,"{0:.2f}".format(act_h),(x+10,int((y+h+height_water)/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                
                #save the image
                cv2.imwrite("saved/image.jpg",image)
                print("Saved at saved/image.jpg.")
                
                #show the image until q is pressed
                while True:
                    cv2.imshow("frame",image)
                    if cv2.waitKey(5) & 0xFF==ord('q'):
                        break
                cv2.destroyAllWindows()
                
            else:
                print("No water lines were detected.\nTry taking another picture keeping the water line as horizontal as you can.\nIf it still does not work, try changing the lighting conditions.") 
        else:
            print("No bottles were detected.\nTry taking another picture keeping the bottle as vertical as you can.\nIf it still does not work, try changing the lighting conditions.")
    else:
        if choice==1:
            print("Failed to Load Image.\nMake sure your webcam is working.")
            break
        elif choice==2:
            print("Failed to Load Image.\nMake sure the input path is correct.")
        else:
            print("Wrong choice.")
