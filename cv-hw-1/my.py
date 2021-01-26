import os
import cv2 
import numpy as np
import argparse as ap

W = 640
H = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX

fourcc =cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc , 20.0, (W, H) )

r = g = b = np.empty([H, W], dtype = np.uint8) 
for i in range(10):
    b = np.random.randint(0, 255, (H, W), dtype=np.uint8)
    g = np.random.randint(0, 255, (H, W), dtype=np.uint8)
    r = np.random.randint(0, 255, (H, W), dtype=np.uint8)
    __frame = cv2.merge([b, g, r])
    # print(__frame.shape)
    __frame = cv2.putText(__frame, 'HeQingyi 3180105438', (150, 470), FONT, 1, (255, 255, 255), 2)
    out.write(__frame)
    # cv2.imshow('frame', __frame)

r.fill(0)
g.fill(0)
b.fill(0)
__frame = cv2.merge([b, g, r])
__frame = cv2.putText(__frame, 'HeQingyi 3180105438', (150, 470), FONT, 1, (255, 255, 255), 2)

for i in range(5):
    out.write(__frame)

for i in range(36):
    __frame = cv2.ellipse(__frame,(320,300),(120,90),0,0,i*10,(255, (1.0*i/36*255), (1.0*(36-i)/36*255)),-1)
    out.write(__frame)

__frame = cv2.ellipse(__frame,(320,300),(120,90),0,0,360,(221, 233, 239),-1)
pts = np.array([[320, 300],[200, 290],[240,180]], np.int32)
pts = pts.reshape((-1,1,2))
__frame = cv2.fillPoly(__frame,[pts],(221, 233, 239))
out.write(__frame)
out.write(__frame)
out.write(__frame)
pts = np.array([[320, 300],[440, 290],[400,180]], np.int32)
pts = pts.reshape((-1,1,2))
__frame = cv2.fillPoly(__frame,[pts],(221, 233, 239))
out.write(__frame)

__frame = cv2.putText(__frame, 'F', (130, 120), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 3)
out.write(__frame)
__frame = cv2.putText(__frame, 'FL', (130, 120), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 3)
out.write(__frame)
__frame = cv2.putText(__frame, 'FLA', (130, 120), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 3)
out.write(__frame)
__frame = cv2.putText(__frame, 'FLAZ', (130, 120), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 3)
out.write(__frame)
__frame = cv2.putText(__frame, 'FLAZE', (130, 120), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 3)

for i in range(25):
    out.write(__frame)
    # cv2.imshow('frame', __frame)

# -------------------------------------------------------------
parser = ap.ArgumentParser()
parser.add_argument("path", help = "path include the input(images and a video)")
PATH = parser.parse_args().path

# -------------------------------------------------------------
frame = __frame
PATHVIDEO = []
for maindir, subdir, file_name_list in os.walk(PATH):
    for filename in file_name_list:
        file_name = os.path.splitext(filename)
        pre, suf = file_name
        if suf == '.avi':
            PATHVIDEO.append(filename)
        else :
            print('Processing '+filename)
            frame = __frame
            frame = cv2.merge([b, g, r])
            img = cv2.imread(PATH+"\\"+filename)
            hh, ww, res = img.shape
            rat = min(1.0*H/hh, 1.0*W/ww)
            hh = int(hh * rat)
            ww = int(ww * rat)
            img = cv2.resize(img , (ww, hh), interpolation=cv2.INTER_NEAREST)
            pos_h = (H-hh)//2
            pos_w = (W-ww)//2
            frame[pos_h:pos_h+hh, pos_w:pos_w+ww] = img
            
            frame = cv2.putText(frame, 'HeQingyi 3180105438', (150, 470), FONT, 1, (255, 255, 255), 2)
            # cv2.imshow('frame',frame)
            for i in range(7):
                tmp = cv2.addWeighted(frame, 1.0*i/7, __frame, 1.0*(7-i)/7, 0.0)
                out.write(tmp)

            for i in range(10):
                out.write(frame)
            __frame = frame
            
for i in range(len(PATHVIDEO)):
    cap = cv2.VideoCapture(PATH+"\\"+PATHVIDEO[i])
    print('Processing '+PATHVIDEO[i])
    while (True):
        ret, frame = cap.read()
        if frame is None :
            break
        img = frame
        frame = cv2.merge([b, g, r])
        hh, ww, res = img.shape
        rat = min(1.0*H/hh, 1.0*W/ww)
        hh = int(hh * rat)
        ww = int(ww * rat)
        img = cv2.resize(img , (ww, hh), interpolation=cv2.INTER_NEAREST)
        pos_h = (H-hh)//2
        pos_w = (W-ww)//2
        frame[pos_h:pos_h+hh, pos_w:pos_w+ww] = img
        frame = cv2.putText(frame, 'HeQingyi 3180105438', (150, 470), FONT, 1, (255, 255, 255), 2)
        
        out.write(frame)
        # cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break 
        __frame = frame
            
cap.release()
out.release()
cv2.destroyAllWindows()

print('-------------- Work Done --------------')
