#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

enum {
    EVEN, ODD
};


inline void thinningIteration(cv::Mat& img, int iter) {
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_32F);
    for (int i = 1; i < img.rows-1; i++) {
        for (int j = 1; j < img.cols-1; j++) {
           float *val = new float[9];
           int icounter = 0;
           for (int y = -1; y <= 1; y++) {
              for (int x = -1; x <= 1; x++) {
                 val[icounter] = img.at<float>(i + y, j + x);
                 icounter++;
              }
           }
           int A = ((val[1] == 0 && val[2] == 1) ? ODD : EVEN)
              + ((val[2] == 0 && val[5] == 1) ? ODD : EVEN)
              + ((val[5] == 0 && val[8] == 1) ? ODD : EVEN)
              + ((val[8] == 0 && val[7] == 1) ? ODD : EVEN)
              + ((val[7] == 0 && val[6] == 1) ? ODD : EVEN)
              + ((val[6] == 0 && val[3] == 1) ? ODD : EVEN)
              + ((val[3] == 0 && val[0] == 1) ? ODD : EVEN)
              + ((val[0] == 0 && val[1] == 1) ? ODD : EVEN);
            int B  = val[0] + val[1] + val[2] + val[3]
               + val[5] + val[6] + val[7] + val[8];
            int m1 = iter == EVEN ? (val[1] * val[5] * val[7])
               : (val[1] * val[3] * val[5]);
            int m2 = iter == EVEN ? (val[3] * val[5] * val[7])
               : (val[1] * val[3] * val[7]);
            if (A == 1 && (B >= 2 && B <= 6) && !m1 && !m2) {
               marker.at<float>(i, j) = sizeof(char);
            }
            free(val);
        }
    }
    cv::bitwise_not(marker, marker);
    cv::bitwise_and(img, marker, img);
}


inline void thinning(cv::Mat& image) {
   
    if (image.type() == CV_8UC3) {
       cv::cvtColor(image, image, CV_BGR2GRAY);
    }
    cv::Mat img;
    image.convertTo(img, CV_32F, 1/255.0);
    
    cv::Mat prev = cv::Mat::zeros(img.size(), CV_32F);
    cv::Mat difference;

    do {
        thinningIteration(img, 0);
        thinningIteration(img, 1);
        cv::absdiff(img, prev, difference);
        img.copyTo(prev);
    } while (cv::countNonZero(difference) > 0);
    
    cv::imshow("thinning", img);
    cv::imshow("Original", image);
    cv::waitKey(0);
}
