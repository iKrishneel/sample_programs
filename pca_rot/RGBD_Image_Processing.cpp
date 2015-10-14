//
//  Color Histogram Descriptors.cpp
//  Handheld Object Tracking
//
//  Created by Chaudhary Krishneel on 11/11/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.
//

#include "RGBD_Image_Processing.h"

/**
 * Function to extract the lines
 */
void RGBDImageProcessing::lineDetection(Mat &image, vector<Vec2f> &s_lines, double langle, double hangle)
{
   vector<Vec2f> lines;
   Mat dst;
   this->extractImageContour(image, dst, true);

   //! detect the lines
   HoughLines(dst, lines, sizeof(char), CV_PI/180, 100);

   cvtColor(image, image, CV_GRAY2BGR);
   
   for (size_t i = 0; i < lines.size(); i++) {
      float rho = lines[i][0];
      float theta = lines[i][1];
      
      if(theta > (CV_PI / 180.0 * hangle) || theta < (CV_PI / 180.0 * langle))
      {
         Point pt1, pt2;

         double a = cos(theta);
         double b = sin(theta);
         double x0 = a * rho;
         double y0 = b * rho;

         pt1.x = cvRound (x0 + 1000 * (-b));
         pt1.y = cvRound (y0 + 1000 * (a));
         pt2.x = cvRound (x0 - 1000 * (-b));
         pt2.y = cvRound (y0 - 1000 * (a));
         line(image, pt1, pt2, Scalar(0,255,0), 2, CV_AA);

         s_lines.push_back(lines[i]);
      }
   }
   imshow("Hough Line", image);
}

/**
 * Function to extract the image contour
 */
void RGBDImageProcessing::extractImageContour(Mat &image,
                                              Mat &canny_out,
                                              bool isContour,
                                              int contour_thresh)
{
   if(!image.data)
   {
      //ROS_ERROR("No Input Image Found");
   }
   else
   {
      vector<Vec4i> hierarchy;

      //GaussianBlur(image, image, Size(7, 7), 0);
      Mat img = image.clone();
      
      if(img.type() != CV_8UC1)
      {
         cvtColor(img, img, CV_BGR2GRAY);
      }
      Canny(img, canny_out, 5, 50, 5, true);
      
      if(isContour)
      {
         std::vector<std::vector<cv::Point> > _contour;
         
         Mat contImg;// = cv::Mat::zeros(image.size(), CV_8U);
         cvtColor(image, contImg, CV_GRAY2BGR);
         findContours(canny_out,
                      _contour,
                      hierarchy,
                      CV_RETR_LIST,
                      CV_CHAIN_APPROX_TC89_KCOS,
                      Point(0, 0));
         for (int i = 0; i < _contour.size(); i++)
         {
            if(cv::contourArea(_contour[i]) > contour_thresh)
            {
               this->contours.push_back(_contour[i]);
               drawContours(contImg,
                            _contour,
                            i,
                            Scalar(0, 255, 0),
                            1,
                            8,
                            hierarchy,
                            0,
                            Point());
            }
            else
            {
               //! #NOTE CHANGE THE ALGORITHM TO COPY ON CONTOUR REGION
               drawContours(image,
                            _contour,
                            i,
                            Scalar(0, 0, 0),
                            CV_FILLED,
                            8,
                            hierarchy,
                            0,
                            Point());
            }
            //imshow("Contours", contImg);
            //waitKey();
         }
         imshow("Contours", contImg);
         std::cout << "Contour Size: " << this->contours.size() << std::endl;
         //imshow("Canny Image", canny_out);
      }
   }
}


/**
 * Function to compute image line extraction on the roi
 */
void RGBDImageProcessing::roiLineDetection(Mat &image, Rect rect)
{
   if(image.empty())
   {
      //ROS_ERROR("No image to extract ROI line detection");
      return;
   }
   Mat roi = image(rect);
   //lineDetection(roi);

   imshow("roi line", roi);
}


void RGBDImageProcessing::getContour(std::vector<std::vector<cv::Point> > &img_contour)
{
   img_contour.clear();
   img_contour = this->contours;
}
