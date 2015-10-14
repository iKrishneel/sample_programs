
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <omp.h>

#include <iostream>
#include <string>

cv::Mat readImage(std::string path) {
    cv::Mat image = cv::imread(path, 1);
    if (image.empty()) {
       std::cout << "NO IMAGE" << std::endl;
       exit(-1);
    }
    return image;
}

double likelihood(double distance, double w) {
    
    return (1.0 / (1 + w *(std::pow(distance/255, 2))));
}


double classifyPixel(const cv::Vec3b pixel, const cv::Mat model) {
    if (model.empty()) {
       return 0.0;
    }
    double max_likelihood = 0;
#pragma omp parallel for shared(max_likelihood) collapse(2)
    for (int j = 0; j < model.rows; j++) {
       for (int i = 0; i < model.cols; i++) {
          double m = 0.0;
          for (int k = 0; k < 3; k++) {
             double v = likelihood(pixel.val[k] - model.at<cv::Vec3b>(
                                      j, i)[k], 0.50);
             m += v;
          }
          if (m/3 > max_likelihood) {
             max_likelihood = m;
          }
       }
    }
    return max_likelihood;
}


int main(int argc, char *argv[]) {

    cv::Mat model = readImage(argv[1]);
    cv::Mat image = readImage(argv[2]);

    cv::Mat probablity = cv::Mat::zeros(image.size(), CV_8UC3);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          double p = classifyPixel(image.at<cv::Vec3b>(j, i), model);
          probablity.at<cv::Vec3b>(j, i)[0] = p * 255;
          probablity.at<cv::Vec3b>(j, i)[1] = p * 255;
          probablity.at<cv::Vec3b>(j, i)[2] = p * 255;
       }
    }

    cv::imshow("probability-map", probablity);
    cv::waitKey(0);
    
    return 0;
}

