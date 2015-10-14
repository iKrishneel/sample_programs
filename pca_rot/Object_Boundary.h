//
//  Object Boundary.h
//  Handheld Object Tracking
//
//  Created by Chaudhary Krishneel on 3/11/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.
//

#ifndef __Handheld_Object_Tracking__Object_Boundary__
#define __Handheld_Object_Tracking__Object_Boundary__

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

typedef struct cvMouseParam{
    
    Point x;
    Point y;
    const char* winName;
    Mat image;
    
}cvMouseParam;

class ObjectBoundary {

private:
    /**
     * Mouse callback function that allows user to specify the
     * initial object region.
     * Parameters are as specified in OpenCV documentation.
     */
    static void cvMouseCallback(int event, int x, int y, int flags, void* param){
        
        cvMouseParam *p = (cvMouseParam*)param;
        Mat clone;
        static int pressed = false;
        
        if (!p->image.data) {
            cout << "No Image Found in MouseCallback" << endl;
            return;
        }
    
        if (event == CV_EVENT_LBUTTONDOWN) {
            p->x.x = x;
            p->x.y = y;
            pressed = true;
        }
        //! on left button press, remember first corner of rectangle around object
        else if (event == CV_EVENT_LBUTTONUP){
            p->y.x = x;
            p->y.y = y;
            clone = p->image.clone();
            rectangle(clone, p->x, p->y, Scalar(0,255,0), 2);
            imshow(p->winName, clone);
//            cvDestroyWindow(p->winName);
            pressed = false;
        }
        //! on mouse move with left button down, draw rectangle
        else if (event == CV_EVENT_MOUSEMOVE && flags & CV_EVENT_FLAG_LBUTTON){
            clone = p->image.clone();
            rectangle(clone, p->x, p->y, Scalar(0,255,0),2);
            imshow(p->winName, clone);
//            cvDestroyWindow(p->winName);
        }
    }
    cvMouseParam *mouseParam;
    
public:
    Rect cvGetobjectBoundary(const Mat &);

    ObjectBoundary();
};
#endif /* defined(__Handheld_Object_Tracking__Object_Boundary__) */
