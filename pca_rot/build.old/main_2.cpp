#include <iostream>
#include <math.h>
#include <fstream>
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
bool getObjectROI(Mat &);
void computeImageGradient(Mat &);
void colorFilterTrackerBar(Mat &);
void colorFilter(Mat &, cv::Scalar, cv::Scalar, bool = false);
void getImageContours(Mat &, int = 30);
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

   bool is_roi = getObjectROI(image);
   if(!is_roi)
   {
      std::cout << "Selected Object Size is too small to be processed...." << std::endl;
      exit(-1);
   }
   
   
   Mat img_bw;
   //cvtColor(image, img_bw, CV_BGR2GRAY);

   colorFilter(image, colorRangeMIN, colorRangeMAX, false);
   //colorFilterTrackerBar(image);
   cvtColor(image, img_bw, CV_BGR2GRAY);

   std::clock_t start;
   double duration;
   start = std::clock();
   
   computeImageGradient(img_bw);

   duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
   std::cout<<"Total Processing Time: "<< duration <<'\n';

   
   Mat c_out;
   Mat img_gray;
   //cv::threshold(img_bw, img_gray, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
   cvtColor(image, image, CV_BGR2GRAY);
   //img_proc->extractImageContour(image, c_out, true);   
    
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
 * Function to estimate the orientation of an object region using PCA
 * analysis on the object region
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
 * Function to compute the raw image gradient using sobel operation in
 * x/y direction
 */
void computeImageGradient(Mat &image)
{
   GaussianBlur(image, image, Size(5,5), 1);
   Mat src_gray = image.clone();
   int scale = 1;
   int delta = 0;
   int ddepth = CV_16S;
   Mat grad_x, grad_y;
   Mat grad;
   Mat abs_grad_x, abs_grad_y;

   Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
   convertScaleAbs( grad_x, abs_grad_x );
   Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
   convertScaleAbs( grad_y, abs_grad_y );
   addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
   grad = grad * 2;
   imageMorphologicalOp(grad);

   imshow( "gradient", grad );
   waitKey();
}

/**
 * User defined object region via bounding box plot
 */
bool getObjectROI(Mat &image)
{
   ObjectBoundary *obj_roi = new ObjectBoundary();
   Rect rect = obj_roi->cvGetobjectBoundary(image);

   if(rect.width < 10 && rect.height < 10)
   {
      return false;
   }
   else
   {
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
      return true;
   }
}


/**
 * Function to filter the input image using the pre-set object color filter
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
 * Function to allow user to update the color thresholding
 */
int min_valR = 0;
int max_valR = 255;

int min_valG = 0;
int max_valG = 255;

int min_valB = 0;
int max_valB = 255;

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
void computeEdgeCurvatureOrientation(std::vector<std::vector<cv::Point> > &contours_tangent,
                                     std::vector<std::vector<cv::Point> > &contours,
                                     std::vector<std::vector<float> > &orientation_)
{
   for (int j = 0; j < contours_tangent.size(); j++) {
      std::vector<float> ang;
      for (int i = 0; i < contours_tangent[j].size(); i++) {
         float angle_ = 0.0f;
         if( contours_tangent[j][i].x == 0)
         {
            angle_ = 90.0f;
         }
         else
         {
            angle_ = atan2(contours_tangent[j][i].y, contours_tangent[j][i].x) * (180/CV_PI);
         }
         ang.push_back(static_cast<float>(angle_));
      }
      orientation_.push_back(ang);
      ang.clear();
   }
}


/**
 * Compute the image pixel curvature on the pixel edges. 
 * "http://en.wikipedia.org/wiki/Finite_difference#Forward.2C_backward.2C_and_central_differences""
 */
void computeEdgeCurvature(std::vector<std::vector<cv::Point> > &contours,
                          std::vector<std::vector<cv::Point> > &tangents,
                          std::vector<std::vector<float> > &orientation_,
                          std::vector<cv::Point> &hcurvature_pts)
{
   for (int j = 0; j < contours.size(); j++) {
      std::vector<cv::Point> tangent;
      
      /* estimate the first tangent line*/
      cv::Point2f edge_pt = contours[j].front();
      cv::Point2f edge_tngt = contours[j].back() - contours[j].at(1);
      tangent.push_back(edge_tngt);

      const int neighbor_pts = 0;
      if(contours[j].size() > sizeof(short))
      {
         for (int i = sizeof(char) + neighbor_pts;
              i < contours[j].size() - sizeof(char) - neighbor_pts;
              i++)
         {
            edge_pt = contours[j][i];
            edge_tngt = contours[j][i-1-neighbor_pts] - contours[j][i+1+neighbor_pts];
            tangent.push_back(edge_tngt);
         }
      }
      tangents.push_back(tangent);
   }
   
   /* compute the tangent orientation */
   computeEdgeCurvatureOrientation(tangents, contours, orientation_);

   
   for (int j = 0; j < tangents.size(); j++) {
      if(tangents[j].size() < 2)
      {
         continue;
      }
      for (int i = sizeof(char); i < tangents[j].size() - sizeof(char); i++) {

         float y1 = tangents[j][i+1].y;
         float x1 = tangents[j][i+1].x;
         
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
         if((tang1 >= 0.0f  && tang0 < 0.0f) || (tang1 < 0.0f && tang0 >= 0.0f))
         {
            if((abs(tang0 - tang1) < 1.0f) || (abs(tang1 - tang0) < 1.0f))
            {
               continue;
            }
            hcurvature_pts.push_back(contours[j][i]);
            //hcurvature_pts.push_back(contours[j][i-1]);
            
            circle(original_img, contours[j][i], 2, Scalar(255, 0, 0), 1);
            circle(original_img, contours[j][i], 5, Scalar(0,255,0), 2);
            //circle(original_img, contours[j][i-1], 2, Scalar(255, 0, 0), 1);
            //circle(original_img, contours[j][i-1], 5, Scalar(0,255,0), 2);
            
            //circle(image, contours[j][i], 1, Scalar(0,0,255), 2);
            //circle(image, contours[j][i-1], 2, Scalar(0,0,255), 2);
         }
      }
   }
   cv::imshow("tangent", original_img);
}


/**
 * Function to filter multiple points on the same line direction by
 * box filter and comparision of the gradient orientation magnitude
 */
void filterJunctionPoints(cv::Mat &img_bw,
                          std::vector<std::vector<float> > &orientation_,
                          std::vector<cv::Point> &hcurvature_pts)
{
   ofstream outFile;
   outFile.open("hcurve.txt", ios::out);
   for (int i = 0; i < hcurvature_pts.size(); i++) {
      outFile << hcurvature_pts.at(i) << endl;
   }
   outFile.close();
}



/**
 * function to filter points by within some set radius
 */
void filterJunctionPoints(cv::Mat &img_bw,
                          std::vector<cv::Point> &hcurvature_pts,
                          const int radius_ = 10)
{
     // filter very close high curvature region
   for (int i = 0; i < hcurvature_pts.size(); i++) {
      cv::Point cur_pt = hcurvature_pts[i];
      for (int j = 0; j < hcurvature_pts.size(); j++) {
         if(i != j)
         {
            cv::Point nn_pt = hcurvature_pts[j];
            float r = cv::norm(nn_pt - cur_pt);

            if(r <= radius_)
            {
               hcurvature_pts.erase(hcurvature_pts.begin() + j);
            }
         }
      }
   }
   Mat image = img_bw.clone();
   cvtColor(image, image, CV_GRAY2BGR);
   for (int i = 0; i < hcurvature_pts.size(); i++) {
      circle(image, hcurvature_pts[i], 2, Scalar(255,0,255),-1);
   }
   imshow("hc", image);
}


/**
 * Function to estimate the junction in the image silhouette when
 * passed with the normal vector and the high curvature data
 */
void cvJunctionEstimation(Mat &img_bw,
                          std::vector<std::vector<cv::Point> > &contours_,
                          std::vector<std::vector<cv::Point> > &tangents_,
                          std::vector<std::vector<float> > &orientation_,
                          std::vector<cv::Point> &hcurvature_pts)
{
   Mat image = img_bw.clone();
   cvtColor(image, image, CV_GRAY2BGR);
   
   const int threshold_ = sizeof(short);
   const int search_radius = 10;
   
   std::vector<cv::Point> junction_points;
   for(int i = 0; i < hcurvature_pts.size(); i++)
   {
      Mat testImg = Mat::zeros(img_bw.size(), img_bw.type());
      circle(testImg, hcurvature_pts[i], search_radius, Scalar(255), 1);

      Mat mask;
      cv::bitwise_and(img_bw, testImg, mask);
      int whitePix = cv::countNonZero(mask > 0);
      std::cout << "Theshold: " << whitePix << std::endl;
      
      if(whitePix > threshold_ || whitePix == sizeof(char))
      {
         junction_points.push_back(hcurvature_pts[i]);

         if(whitePix == sizeof(char))
         {
            circle(image, hcurvature_pts[i], 2, Scalar(0,255,0),-1);
         }
         else
         {
            circle(image, hcurvature_pts[i], 2, Scalar(255,0,255),-1);
         }
      }
   }
   imshow("junction", image);

   filterJunctionPoints(img_bw, orientation_, junction_points);
   filterJunctionPoints(img_bw, junction_points, search_radius/2);
}

/**
 * Function to perform contour estimation on the silhouette contour
 * and smooth the contour based on the pre-set threshold.
 */
void branchEstimation(Mat &img_bw)
{
   /* get the image contour by fitering the small contour region */
   imshow("input", img_bw);
   cv::Mat canny_out;
   int contour_area_thresh = 50;
   img_proc->extractImageContour(img_bw, canny_out, true, contour_area_thresh);
   std::vector<std::vector<cv::Point> > contours;
   img_proc->getContour(contours);

   // compute the edge orientation and high curvature points
   std::vector<std::vector<cv::Point> > tangents;
   std::vector<std::vector<float> > orientation_;
   std::vector<cv::Point> hcurvature_pts;
   computeEdgeCurvature(contours, tangents, orientation_, hcurvature_pts);

   // junction estimation
   cvJunctionEstimation(img_bw, contours, tangents, orientation_, hcurvature_pts);

   cv::imshow("skeleton", img_bw);
   cv::waitKey();
}

/**
 * Adequate image smoothing 
 */
void imageMorphologicalOp(cv::Mat &src, bool is_errode)
{
   cv::Mat erosion_dst;
   int erosion_size = 5;
   int erosion_const = 2;
   
   int erosion_type = MORPH_ELLIPSE;
   cv::Mat element = cv::getStructuringElement(erosion_type,
                                               cv::Size(erosion_const * erosion_size + 1,
                                                        erosion_const * erosion_size + 1),
                                               cv::Point(erosion_size, erosion_size ));

   const int smooth_window = 7;
   const int smooth_sigma = 0;
   cv::GaussianBlur(src, src,
                    cv::Size(smooth_window, smooth_window),
                    smooth_sigma);
   cv::erode( src, erosion_dst, element );
   //erosion_dst = erosion_dst * 4;
   
   /*convert the eroded image to binary*/
   Mat img_bw;
   cvCreateBinaryImage(erosion_dst, img_bw);   
   thinning(img_bw);
       
   /* estimate the branch */
   branchEstimation(img_bw);

   imshow("image", src);
   imshow( "Erosion Demo", erosion_dst );
   imshow("Binary", img_bw);
   waitKey();
}
