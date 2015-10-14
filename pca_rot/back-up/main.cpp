#include <iostream>
#include <math.h>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "Object_Boundary.h"
#include "RGBD_Image_Processing.h"
#include "contour_thinning.h"

using namespace std;
using namespace cv;

const cv::Scalar colorRangeMIN = cv::Scalar(0, 56, 5);
const cv::Scalar colorRangeMAX = cv::Scalar(64, 119, 103);

double getOrientation(vector<Point> &, Mat &);
void getObjectROI(Mat &);
void computeImageGradient(Mat &);
void colorFilterTrackerBar(Mat &);
void colorFilter(Mat &, cv::Scalar, cv::Scalar, bool = false);
void getImageContours(Mat &, int = 30);

void estimateDOGEdge(Mat &);
void imageMorphologicalOp(Mat &, bool = true);


RGBDImageProcessing * img_proc = new RGBDImageProcessing();

Mat original_img;
int main(int argc, char *argv[])
{
   Mat image;
   
   if(argc < 2)
   {
      image = imread("../pca_test1.jpg", 1);
   }
   else
   {
      image = imread(argv[1], 1);
   }

   if(image.empty())
   {
      std::cout << "No Image Found...." << std::endl;
      return -1;
   }
   const int downsample_ = 2;
   cv::resize(image, image, cv::Size(image.cols/downsample_, image.rows/downsample_));
   
   original_img = image.clone();

   getObjectROI(image);
   Mat img_bw;
   //cvtColor(image, img_bw, CV_BGR2GRAY);

   colorFilter(image, colorRangeMIN, colorRangeMAX, false);
   //colorFilterTrackerBar(image);
   cvtColor(image, img_bw, CV_BGR2GRAY);

   //estimateDOGEdge(img_bw);   
   computeImageGradient(img_bw);
   
   //img_proc->lineDetection(image);

   Mat c_out;
   Mat img_gray;
   //cv::threshold(img_bw, img_gray, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
   cvtColor(image, image, CV_BGR2GRAY);
   //img_proc->extractImageContour(image, c_out, true);


   /* Dilate and errode*/

   
   //getImageContours(original_img);
   
    
   vector<Vec4i> lines;
   cv::HoughLinesP(img_bw, lines, 1, CV_PI/180 , 20, 5, 5);
 
   imshow("input", image);
   waitKey();
   
   
   //! Find all the contours in the thresholded image
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;
   findContours(img_bw, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

   Mat img = image.clone();
   for (size_t i = 0; i < contours.size(); ++i)
   {
      //! Calculate the area of each contour
      double area = contourArea(contours[i]);
      //! Ignore contours that are too small or too large
      if (area < 1e2 || 1e5 < area)
      {
         continue;
      }
      else
      {
         //! Draw each contour only for visualisation purposes
         drawContours(img, contours, i, CV_RGB(255, 0, 0), 2, 8, hierarchy, 0);
         //! Find the orientation of each shape
         getOrientation(contours[i], img);
      }
   }  
   cv::imshow("img_bw", img_bw);
   cv::imshow("orig", img);
   cv::waitKey(0);
   
   return 0;
}

/**
 * 
 */
double getOrientation(vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    Mat data_pts = Mat(pts.size(), 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
 
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
 
    //Store the position of the object
    Point pos = Point(pca_analysis.mean.at<double>(0, 0),
                      pca_analysis.mean.at<double>(0, 1));
 
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
 
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
 
    // Draw the principal components
    circle(img, pos, 3, CV_RGB(255, 0, 255), 2);
    line(img, pos, pos + 0.02 * Point(eigen_vecs[0].x * eigen_val[0],
                                      eigen_vecs[0].y * eigen_val[0]) ,
         CV_RGB(0, 255, 0));
    line(img, pos, pos + 0.02 * Point(eigen_vecs[1].x * eigen_val[1],
                                      eigen_vecs[1].y * eigen_val[1]) ,
         CV_RGB(0, 0, 255));
    return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}


/**
 * 
 */
void computeImageGradient(Mat &image)
{
   GaussianBlur(image, image, Size(5,5), 1);

   Mat src_gray = image.clone();
   /*
   Mat xsobel, ysobel;
    Sobel(image, xsobel, CV_32F, 1, 0, 5);
    Sobel(image, ysobel, CV_32F, 0, 1, 1);
    
    Mat Imag, Iang; //! Angle and Maginitue
    cartToPolar(xsobel, ysobel, Imag, Iang, true);

    normalize(Imag, Imag, 0, 1, NORM_MINMAX, -1, Mat());
    
    const int angle_ = 360;
    add(Iang, Scalar(angle_), Iang, Iang < angle_);
    add(Iang, Scalar(-angle_), Iang, Iang >= angle_);


    normalize(Iang, Iang, 0, 1, NORM_MINMAX, -1, Mat());
    
    Mat cb;
    multiply(Imag, Iang, cb);
    imshow("Mag", Imag);
    imshow("ang", Iang);
    imshow("weight", cb);
*/
   int scale = 1;
   int delta = 0;
   int ddepth = CV_16S;

   /// Generate grad_x and grad_y
   Mat grad_x, grad_y;
   Mat grad;
   Mat abs_grad_x, abs_grad_y;
  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );
  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  grad = grad * 2;
  
  //getImageContours(grad);
  imageMorphologicalOp(grad);
  
  imshow( "gradient", grad );
    waitKey();
}

/**
 * User defined object region via bounding box plot
 */
void getObjectROI(Mat &image)
{
   ObjectBoundary *obj_roi = new ObjectBoundary();
   Rect rect = obj_roi->cvGetobjectBoundary(image);

   Mat img = Mat::zeros(image.rows, image.cols, image.type());
   for (int j = 0; j < image.rows; j++) {
      for (int i = 0; i < image.cols; i++) {
         if(i > (rect.x - 1) && i < (rect.x + rect.width + 1) &&
            j > (rect.y - 1) && j < (rect.y + rect.height + 1))
         {
            img.at<Vec3b>(j,i) = image.at<Vec3b>(j,i);
         }
      }
   }
   image = img.clone();
}


/**
 * 
 */
void colorFilter(Mat &image, Scalar colorMin, Scalar colorMax, bool isFind)
{   if(isFind)
   {
      colorFilterTrackerBar(image);
   }
   else
   {
      Mat dst;
      cv::inRange(image, colorMin, colorMax, dst);

      //cv::bitwise_and(img, dst, outimg);
      for (int j = 0; j < image.rows; j++) {
         for (int i = 0; i < image.cols; i++) {
            if((float)dst.at<uchar>(j,i) == 0)
            {
               image.at<Vec3b>(j,i)[0] = 0;
               image.at<Vec3b>(j,i)[1] = 0;
               image.at<Vec3b>(j,i)[2] = 0;
            }
         }
      }
   }
}



/**
 * 
 */
int min_valR = 0;
int max_valR = 256;

int min_valG = 0;
int max_valG = 256;

int min_valB = 0;
int max_valB = 256;

Mat dst;
Mat img;

string window_name = "Result";
void threshCallBack(int, void*)
{
   dst = img.clone();
   Scalar lowerb = Scalar(min_valB, min_valG, min_valR);
   Scalar upperb = Scalar(max_valB, max_valG, max_valR);
   cv::inRange(img, lowerb, upperb, dst);

   Mat outimg = img.clone();
   //cv::bitwise_and(img, dst, outimg);
   for (int j = 0; j < img.rows; j++) {
      for (int i = 0; i < img.cols; i++) {
         if((float)dst.at<uchar>(j,i) == 0)
         {
            outimg.at<Vec3b>(j,i)[0] = 0;
            outimg.at<Vec3b>(j,i)[1] = 0;
            outimg.at<Vec3b>(j,i)[2] = 0;
         }
      }
   }

   
   imshow( window_name, dst );
   imshow("out", outimg);
}

void colorFilterTrackerBar(Mat &image)
{
   img = image.clone();
   namedWindow("Result", CV_WINDOW_AUTOSIZE);
   createTrackbar("Red Min", window_name, &min_valR, 256);
   createTrackbar("Red Max", window_name, &max_valR, 256);
   
   createTrackbar("Green Min", window_name, &min_valG, 256);
   createTrackbar("Green Max", window_name, &max_valG, 256);
    
   createTrackbar("Blue Min", window_name, &min_valB, 256);
   createTrackbar("Blue Max", window_name, &max_valB, 256);
   

   //cvtColor(image, img, CV_BGR2HSV);
   int c;
   while(true)
   {
      threshCallBack(0,0);
      c = waitKey( 20 );
      if( (char)c == 27 )
      {
         break;
      }
   }
}


/**
 * estimate the image contours
 */
void getImageContours(Mat &image, int contour_thresh)
{
   if(image.empty() || image.type() != CV_8UC1)
   {
      std::cout << "Image is empty" << std::endl;
      return;
   }
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;
   findContours(image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

   Mat drawImg;
   cvtColor(image, drawImg, CV_GRAY2BGR);
   
   for (int i = 0; i < contours.size(); i++)
   {
      if(cv::contourArea(contours[i]) > contour_thresh)
      {
         drawContours(drawImg, contours, i, Scalar(255, 0, 255), 2, 8, hierarchy, 0, Point());
      }
   }
   imshow("Image Contours", drawImg);
}


void estimateDOGEdge(Mat &image)
{
   Mat img1, img2;
   GaussianBlur(image, img1, Size(3,3), 0);
   GaussianBlur(image, img2, Size(17,17), 0);

   Mat edgeMap = img1 - img2;

   imshow("Edge", edgeMap);
   waitKey(0);
}





/**
 * Function to create a binary image for any pixel not equal to zero
 */
void cvCreateBinaryImage(Mat image, Mat &img_bw, int lowerb = 0)
{
   img_bw = Mat::zeros(image.size(), CV_8UC1);
   for (size_t j = 0; j < image.rows; j++) {
      for (size_t i = 0; i < image.cols; i++) {
         if(static_cast<int>(image.at<uchar>(j,i)) > lowerb)
         {
            img_bw.at<uchar>(j,i) = 255;
         }
      }
   }
}


/**
 * compute the pixel edge directional orientation 
 */
void computeEdgeTangentOrientation(Mat &image,
                                   std::vector<std::vector<cv::Point> > &contours_tangent,
                                   std::vector<std::vector<cv::Point> > &contours)
{
#define BIN (6)
   const int ANGLE (180/BIN);
   
   std::vector<std::vector<cv::Point> > orientation_bin(BIN);
   for (int j = 0; j < contours_tangent.size(); j++) {
      for (int i = 0; i < contours_tangent[j].size(); i++) {
         /*compute the angle*/
         float angle_ = 0.0f;
         if( contours_tangent[j][i].x == 0)
         {
            angle_ = 90.0f;
         }
         else
         {
            angle_ = atan(contours_tangent[j][i].y/contours_tangent[j][i].x) * (180/CV_PI);
         }
         int w_bin = (static_cast<int>(angle_) / ANGLE) + ((BIN - 1)/2);
         orientation_bin[w_bin].push_back(contours[j][i]);
      }
   }

   for (int j = 0; j < orientation_bin.size(); j++) {
      Mat img = image.clone();//Mat::zeros(image.size(), CV_8UC3);
      for (int i = 1; i < orientation_bin[j].size(); i++) {
         //circle(img, orientation_bin[j][i], 3, Scalar(255,0,255),
         //-1);
         line(img, orientation_bin[j][i-1], orientation_bin[j][i], Scalar(0,255,0), 2);
         
      }
      //imshow("Cluster of lines", img);
      //waitKey();
   }
}




/**
 * Compute the image pixel tanget on the contour. 
 * "http://en.wikipedia.org/wiki/Finite_difference#Forward.2C_backward.2C_and_central_differences""
 */
void computeEdgeTangent(Mat &image, std::vector<std::vector<cv::Point> > &contours)
{
   cvtColor(image, image, CV_GRAY2BGR);
   
   std::vector<std::vector<cv::Point> > tangents;
   for (int j = 0; j < contours.size(); j++) {
      std::vector<cv::Point> tangent;
      
      /* estimate the first tangent line*/
      cv::Point2f edge_pt = contours[j].front();
      cv::Point2f edge_tngt = contours[j].back() - contours[j].at(1);
      tangent.push_back(edge_tngt);
      //cv::line(image, edge_pt + edge_tngt, edge_pt - edge_tngt, cv::Scalar(0,0,255), 2);

      const int neighbor_pts = 0;
      if(contours[j].size() > sizeof(short))
      {
         for (int i = sizeof(char) + neighbor_pts; i < contours[j].size() - sizeof(char) - neighbor_pts; i++) {
            edge_pt = contours[j][i];
            edge_tngt = contours[j][i-1-neighbor_pts] - contours[j][i+1+neighbor_pts];
            tangent.push_back(edge_tngt);
            //cv::line(image, edge_pt + edge_tngt, edge_pt - edge_tngt, cv::Scalar(0,0,255), 2);
         }
      }
      tangents.push_back(tangent);
   }
   /* compute the tangent orientation */
   computeEdgeTangentOrientation(image, tangents, contours);
   
   std::vector<cv::Point> selected_pts; //! memory to hold the point
                                        //! where gradient changes
   const float tangent_range = 0.01f;
   for (int j = 0; j < tangents.size(); j++) {
      if(tangents[j].size() < 2)
      {
         continue;
      }
      for (int i = sizeof(char); i < tangents[j].size(); i++) {

         float y1 = tangents[j][i].y;
         float x1 = tangents[j][i].x;
         
         float y0 = tangents[j][i-1].y;
         float x0 = tangents[j][i-1].x;
         
         float tang1 = 0.0f;
         float tang0 = 0.0f;
         
         if(x1 != 0)
         {
            tang1 = y1/x1;
         }
         if(x0 != 0)
         {
            tang0 = y0/x0;
         }
         //std::cout << "Tanget: " << tang1 << "\tAngle: " << atan2(y1,x1) * 180/CV_PI << std::endl;
         //if(tang1 > -tangent_range && tang1 < tangent_range)
         if((tang1 >= 0.0f  && tang0 < 0.0f) || (tang1 < 0.0f && tang0 >= 0.0f))
         {
            selected_pts.push_back(contours[j][i]);
            //selected_pts.push_back(contours[j][i-1]);
            
            circle(original_img, contours[j][i], 2, Scalar(255, 0, 0), 1);
            circle(original_img, contours[j][i], 5, Scalar(0,255,0), 2);
            circle(original_img, contours[j][i-1], 2, Scalar(255, 0, 0), 1);
            circle(original_img, contours[j][i-1], 5, Scalar(0,255,0), 2);
/*
            circle(image, contours[j][i], 2, Scalar(255, 0, 0), -1);
            circle(image, contours[j][i], 5, Scalar(0,0,255), 2);
            circle(image, contours[j][i-1], 2, Scalar(255, 0, 0), -1);
            circle(image, contours[j][i-1], 5, Scalar(0,0,255), 2);
*/
            cv::line(image, contours[j][i-1], contours[j][i], Scalar(0,255,0), 2);        
         }
      }
      //cv::imshow("masked", image);
      //waitKey();
   }
   cv::imshow("masked", image);
   cv::imshow("tangent", original_img);
   cv::waitKey();
}

/**
 * 
 */
void branchEstimation(Mat &img_bw)
{
   /* get the image contour by fitering the small contour region */
   cv::Mat canny_out;
   int contour_area_thresh = 100;
   img_proc->extractImageContour(img_bw, canny_out, true, contour_area_thresh);
   std::vector<std::vector<cv::Point> > contours;
   img_proc->getContour(contours);
   
   computeEdgeTangent(img_bw, contours);
}


/**
 * Adequate image smoothing 
 */
void imageMorphologicalOp(cv::Mat &src, bool is_errode)
{
   cv::Mat erosion_dst;
   int erosion_size = 5;
   int erosion_const = 2;

   cv::Mat dilation_dst;
   int dilation_size = 1;
   int dilation_const = 2;

   int erosion_type = MORPH_ELLIPSE;
   cv::Mat element = cv::getStructuringElement(erosion_type,
                                               cv::Size(erosion_const * erosion_size + 1,
                                                        erosion_const * erosion_size + 1),
                                               cv::Point(erosion_size, erosion_size ));

   cv::GaussianBlur(src, src, cv::Size(7, 7), 0);
   cv::erode( src, erosion_dst, element );
   //erosion_dst = erosion_dst * 4;

   /* dilate the image by same factor as erosion*/
   int dilation_type = MORPH_ELLIPSE;
   cv::Mat d_element = cv::getStructuringElement(dilation_type,
                                                 cv::Size(dilation_const * dilation_size + 1,
                                                          dilation_const * dilation_size + 1),
                                                 cv::Point(dilation_size, dilation_size));
   //dilate( erosion_dst, dilation_dst, d_element );
   //erosion_dst = dilation_dst;
   
   /*convert the eroded image to binary*/
   Mat img_bw;
   cvCreateBinaryImage(erosion_dst, img_bw);

   std::clock_t start;
   double duration;
   start = std::clock();
   
   cv::Mat bw = img_bw.clone();
   thinning(bw);
   cv::imshow("dst", bw);

   duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
   std::cout << "Time " << duration << std::endl;
   waitKey();
    
   
   /* estimate the branch */
   branchEstimation(img_bw);

   erosion_const = 3;
   element = cv::getStructuringElement(erosion_type,
                                               cv::Size(erosion_const * erosion_size + 1,
                                                        erosion_const * erosion_size + 1),
                                               cv::Point(erosion_size, erosion_size ));
   //cv::erode(img_bw, img_bw, element );
   

   //imshow("lines", img_lines);
   imshow("image", src);
   imshow( "Erosion Demo", erosion_dst );
   imshow("Binary", img_bw);
   waitKey();
}
