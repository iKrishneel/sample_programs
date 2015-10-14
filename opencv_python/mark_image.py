#!/usr/bin/env python

import numpy as np
import cv2

is_drawing = False
mode = True
ix, iy = -1, -1
img= np.zeros((512,512,3), np.uint8)

def mouse_event(event, x, y, flags, param):
    global ix, iy, drawing_mode
    global is_drawing
    drawing = is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        print 'Drawing'
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            print 'FINISH', ix, iy
    
def read_images(path_text):
    image_lists = []
    with open(path_text, 'r') as f:
        for line in f:
            pth = line.split()
            image = cv2.imread(pth[0])
            image_lists.append(image)
    return image_lists


def draw_bounding_box(image_lists):
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_event)

    global img
    img = image_lists[0]
    cv2.imshow('image',img)
    #for i in image_lists:
        #cv2.imshow('image', i)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def main():
    image_lists = read_images('text.txt')
    draw_bounding_box(image_lists)
    

if __name__ == "__main__":
    main()

