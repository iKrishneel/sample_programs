//
//  RGBD_Image_Processing.h
//  boundary_estimation
//
//  Created by Chaudhary Krishneel on 3/11/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.
//

#ifndef __boundary_estimation__RGBD_Image_Processing__
#define __boundary_estimation__RGBD_Image_Processing__

#include <iostream>
   
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class RGBDImageProcessing
{
private:
   
   //! downsampling scale size
   int downSample;

   //! Path to the template
   string template_path__;
   
   //! memeory to hold the region of the detected object
   vector<Rect> object_region__;
   vector<vector<Point> > contours;
   
public:
   
   RGBDImageProcessing(int dw_sm = sizeof(char), const int thresh_ = 4) :
      downSample(dw_sm)
   {
      object_region__.clear();
   }
   
   void extractImageContour(Mat &, Mat &, bool = false, int = 30);
   void lineDetection(Mat &, vector<Vec2f> &, double = 20, double = 160);
   void roiLineDetection(Mat &, Rect);
   void getContour(std::vector<std::vector<cv::Point> > &);
};


#endif /* defined(__boundary_estimation__RGBD_Image_Processing__) */
