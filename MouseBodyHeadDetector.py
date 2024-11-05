import torch
import pandas as pd
import cv2

def getMouseCoordinates(model, img):
    results = model(img)
    bodydf = results.pandas().xyxy[0]
    bodydf = bodydf[bodydf['name'].str.contains('mouse')]
    bodydf.sort_values(by=['confidence'], ascending=False)
    bodydf = bodydf.reset_index(drop=True)
    headdf = results.pandas().xyxy[0]
    headdf = headdf[headdf['name'].str.contains('heads')]
    headdf.sort_values(by=['confidence'], ascending=False)
    headdf = headdf.reset_index(drop=True)
    return(bodydf, headdf)

if __name__ == "__main__":
    model = torch.hub.load('./yolov5', 'custom', path='./mouse_head_body_v19.pt', source='local')
    model.conf = 0.700
    model.iou = 0.700 
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread("test.jpg")
    bodydf, headdf = getMouseCoordinates(model, img)
    bodyxmin = round(bodydf['xmin'][0])
    bodyxmax = round(bodydf['xmax'][0])
    bodyymin = round(bodydf['ymin'][0])
    bodyymax = round(bodydf['ymax'][0])            
    bodyx = round(((bodyxmax - bodyxmin)/2)+bodyxmin)
    bodyy = round(((bodyymax - bodyymin)/2)+bodyymin)
    xmin = round(headdf['xmin'][0])
    xmax = round(headdf['xmax'][0])
    ymin = round(headdf['ymin'][0])
    ymax = round(headdf['ymax'][0])
    headx = round(((xmax - xmin)/2)+xmin)
    heady = round(((ymax - ymin)/2)+ymin)
    cX = round(((bodyxmax - bodyxmin)/2)+bodyxmin)
    cY = round(((bodyymax - bodyymin)/2)+bodyymin)
    headcenterX = round(((xmax - xmin)/2)+xmin)
    headcenterY = round(((ymax - ymin)/2)+ymin)
    img = cv2.line(img, ((cX),(cY)), ((headcenterX),(headcenterY)), (0, 255, 0), thickness)
    img = cv2.rectangle(img, (bodyxmin,bodyymin),(bodyxmax,bodyymax),(255,0,255),thickness)
    img = cv2.putText(img, 'Body', (bodyxmax,cY), font, 1, (255,0,255), thickness, cv2.LINE_AA)
    img = cv2.rectangle(img, (xmin,ymin),(xmax,ymax),(0,255,255),thickness)
    img = cv2.putText(img, 'Head', (xmax,headcenterY), font, 1, (0,255,255), thickness, cv2.LINE_AA)
    img = cv2.rectangle(img, ((headcenterX-thickness),(headcenterY-thickness)), ((headcenterX+thickness),(headcenterY-thickness)), (255,100,100), thickness)
    img = cv2.rectangle(img, ((cX-thickness),(cY-thickness)), ((cX+thickness),(cY+thickness)), (1,0,255), thickness)
    cv2.imshow('Example Result', img)
    cv2.waitKey(0)
