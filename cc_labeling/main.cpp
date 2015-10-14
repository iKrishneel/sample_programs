
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <functional>

using namespace cv;

#include "connected.h"
#include "contour_thinning.h"


/**
 * function to label the binary image using connected component analysis
 */
void cvLabelImageRegion(const cv::Mat &in_img,
                        cv::Mat &labelMD) {
    if (in_img.empty()) {
       std::cout << "Input image Empty!!!" << std::endl;
       return;
    }
    int width = in_img.cols;
    int height = in_img.rows;
    unsigned char *_img = new unsigned char[height*width];
    for (int j = 0; j < height; j++) {
       for (int i = 0; i < width; i++) {
          _img[i + (j * width)] = in_img.at<uchar>(j, i);
       }
    }
    const unsigned char *img = (const unsigned char *)_img;
    unsigned char *out_uc = new unsigned char[width*height];
    ConnectedComponents *connectedComponent =
       new ConnectedComponents(3);
    connectedComponent->connected<
       unsigned char, unsigned char, std::equal_to<unsigned char>, bool>(
       img, out_uc, width, height, std::equal_to<unsigned char> (), false);

    labelMD = cv::Mat(in_img.size(), CV_32F);
    for (int j = 0; j < height; j++) {
       for (int i = 0; i < width; i++) {
          labelMD.at<float>(j, i) = static_cast<float>(
             out_uc[i + (j * width)]);
       }
    }
    free(_img);
    free(out_uc);
    free(connectedComponent);
}

void cvMorphologicalOperations(
    const cv::Mat &img,
    cv::Mat &erosion_dst) {
    if (img.empty()) {
       return;
    }
    int erosion_size = 3;
    int erosion_const = 2;
    int erosion_type = MORPH_ELLIPSE;
    cv::Mat element = cv::getStructuringElement(
       erosion_type,
       cv::Size(erosion_const * erosion_size + sizeof(char),
                erosion_const * erosion_size + sizeof(char)),
       cv::Point(erosion_size, erosion_size));
    cv::dilate(img, erosion_dst, element);
    
    cv::imshow("Errode Image", erosion_dst);
}


/**
 * 
 */
void cvGetImageGrid(
    const cv::Mat &img,
    std::vector<cv::Mat> &img_cells,
    cv::Size cell_size = cv::Size(80, 60)) {
    if (img.empty()) {
       return;
    }
    img_cells.clear();
    Mat img_ = img.clone();
    cvtColor(img_, img_, CV_GRAY2BGR);
    for (int j = 0; j < img.rows; j += cell_size.height) {
       for (int i = 0; i < img.cols; i += cell_size.width) {
          cv::Rect_<int> rect = cv::Rect_<int>(
             i, j, cell_size.width, cell_size.height);
          if (rect.x + rect.width <= img.cols &&
              rect.y + rect.height <= img.rows) {
             cv::Mat roi = img(rect);
             img_cells.push_back(roi);

             cv::rectangle(img_, rect, cv::Scalar(0, 255, 0), 2);
          } else {
             continue;
          }
       }
    }
    cv::imshow("grid map", img_);
}

void colorRegion(const cv::Mat &labelMD, cv::Mat &regionMD) {

    cv::RNG rng(12345);
    cv::Scalar color[100];
    for (int i = 0; i < 100; i++) {
       color[i] = cv::Scalar(
          rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
    
    regionMD = cv::Mat(labelMD.size(), CV_8UC3);
    for (int j = 0; j < labelMD.rows; j++) {
       for (int i = 0; i < labelMD.cols; i++) {
          int lab = (int)labelMD.at<float>(j, i);
          regionMD.at<cv::Vec3b>(j, i)[0] = color[lab].val[0];
          regionMD.at<cv::Vec3b>(j, i)[1] = color[lab].val[1];
          regionMD.at<cv::Vec3b>(j, i)[2] = color[lab].val[2];
       }
    }
}


int main(int argc, char *argv[]) {
    std::clock_t start;
    double duration;
    start = std::clock();
    
    cv::Mat image = cv::imread("../edge.jpg", 0);
    if (image.empty()) {
       std::cout << "NO IMAGE READ..." << std::endl;
    }
    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          if (static_cast<int>(image.at<uchar>(j, i)) > 20) {
             image.at<uchar>(j, i) = 255;
          } else {
             image.at<uchar>(j, i) = 0;
          }
       }
    }
    
    cv::Rect rect(260, 180, 80, 60);
    
    cv::Mat img = image;
    cvMorphologicalOperations(img, img);
    thinning(img);

    std::vector<cv::Mat> img_cells;
    cvGetImageGrid(img, img_cells);

    Mat colorMapMD = Mat(image.size(), CV_8UC3);
    for (int i = 0; i < img_cells.size(); i++) {
       Mat roi = img_cells[i].clone();

       cv::Mat labelMD;
       cvLabelImageRegion(roi, labelMD);

       cv::Mat regionMD;
       colorRegion(labelMD, regionMD);
       cv::imshow("label", regionMD);
       //
    }
    
     duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
     std::cout<<"printf: "<< duration << std::endl;

    cv::cvtColor(image, image, CV_GRAY2BGR);
    cv::rectangle(image, rect, cv::Scalar(0,255,0), 2);
    cv::imshow("image", image);
    cv::waitKey(0);
   
    return 0;
}


