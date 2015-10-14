//
//  Object Boundary.cpp
//  Handheld Object Tracking
//
//  Created by Chaudhary Krishneel on 3/11/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.
//

#include "Object_Boundary.h"

/*
 * Constructor
 */
ObjectBoundary::ObjectBoundary(){
    this->mouseParam = new cvMouseParam();
}

/**
 * Allows the user to interactively select the initial object region
 *
 * @param frame  The frame of video in which objects are to be selected
 * @param region A pointer to an array to be filled with rectangles
 */
Rect ObjectBoundary::cvGetobjectBoundary(const Mat &image){
    
    Mat frame = image.clone();
    this->mouseParam->winName = "Select the Handheld Object Boundary";
    this->mouseParam->image = frame;
    
    cvNamedWindow(this->mouseParam->winName, 1);
    imshow(this->mouseParam->winName, frame);
    cvSetMouseCallback(this->mouseParam->winName, this->cvMouseCallback, this->mouseParam);
    waitKey(0);
    cvDestroyWindow(this->mouseParam->winName);
    
    Rect region;
    region.x = MIN(this->mouseParam->x.x, this->mouseParam->y.x);
    region.y = MIN(this->mouseParam->x.y, this->mouseParam->y.y);
    region.width = MAX(this->mouseParam->x.x, this->mouseParam->y.x) - region.x + sizeof(char);
    region.height = MAX(this->mouseParam->x.y, this->mouseParam->y.y) - region.y + sizeof(char);
    
    return region;
}
