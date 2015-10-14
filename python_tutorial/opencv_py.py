#!/usr/bin/python

import cv2

def webcam():
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        img = cv2.resize(frame, (640, 480))
        img = cv2.flip(img, 1)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    

def main():
    print 'In main..'
    img = cv2.imread("../../man.jpg")
    img = cv2.resize(img, (320, 240))
    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    #main()
    webcam()
