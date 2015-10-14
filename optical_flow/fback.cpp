#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;


static void drawOptFlowMap(
    const Mat& flow, Mat& cflowmap, int step,
    double, const Scalar& color) {
    for (int y = 0; y < cflowmap.rows; y += step)
        for (int x = 0; x < cflowmap.cols; x += step) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            Point pt = Point(x, y);
            Point npt = flow.at<Point2f>(y, x);
            npt.x += pt.x;
            npt.y += pt.y;
            double distance = cv::norm(cv::Mat(pt), cv::Mat(npt));
            if (distance > 10) {
               line(cflowmap, Point(x, y),
                    Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                    color);
               circle(cflowmap, Point(x, y), 2, color, -1);
            }
        }
}

int main(int, char**) {
    Mat prev  = imread("frame0000.jpg");
    Mat frame = imread("frame0001.jpg");

    Mat flow, cflow;
    Mat gray, prevgray, uflow;
    namedWindow("flow", 1);

    cvtColor(prev, prevgray, COLOR_BGR2GRAY);
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
    if (!prevgray.empty()) {
       calcOpticalFlowFarneback(
          prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 1);
       cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
       uflow.copyTo(flow);
       drawOptFlowMap(flow, cflow, 1, 1.5, Scalar(0, 255, 0));
       imshow("flow", cflow);

       // std::cout << uflow << std::endl;
    }
    waitKey(0);
    std::swap(prevgray, gray);
    return 0;
}




