//
//  main.cpp
//  Motion Estimation
//
//  Created by Chaudhary Krishneel on 12/9/13.
//  Copyright (c) 2013 Chaudhary Krishneel. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/videostab/videostab.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <opencv2/videostab/optical_flow.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>

using namespace cv;
using namespace std;
using namespace videostab;

void buildImagePyramid(const Mat &, vector<Mat> &);
void getOpticalFlow(const Mat &, const Mat &,
                    vector<Point2f> &, vector<Point2f> &, vector<uchar> &);
void drawFlowField(Mat &, vector<Point2f> &, vector<Point2f> &);
string float2string(float);

void opticalFlowFromFile();
void computeAffineTransformation(
   const vector<Point2f> &, const vector<Point2f> &);

#define FEATURE_COUNT 500

int main(int argc, const char * argv[]) {

    opticalFlowFromFile();
    return 0;
}



// compute descriptor
void computeKeyPoints(
    vector<Point2f> &points, vector<KeyPoint> &keypoints) {    
    for (int i = 0; i < points.size(); i++) {
       Point2f pt = points[i];
       if (pt.x > 0 && pt.x < 640 && pt.y > 0 && pt.y < 40) {
          KeyPoint kp;
          kp.pt = points[i];
          kp.size = 8.0f;
          keypoints.push_back(kp);
       }
    }
}


void opticalFlowFromFile() {
   
    Mat img1 = imread(
       "frame0000.jpg");
    Mat img2 = imread(
       "frame0001.jpg");

    vector<Point2f> nextPts;
    vector<Point2f> prevPts;
    vector<Point2f> backPts;

    GaussianBlur(img1, img1, Size(1, 1), 1);
    GaussianBlur(img2, img2, Size(1, 1), 1);
    
    Mat gray, grayPrev;
    cvtColor(img1, grayPrev, CV_BGR2GRAY);
    cvtColor(img2, gray, CV_BGR2GRAY);
    // goodFeaturesToTrack(grayPrev, prevPts, FEATURE_COUNT, 0.001,
    // 0.1);
    Ptr<FeatureDetector> detector_h = FeatureDetector::create("FAST");
     Ptr<FeatureDetector> detector_f = FeatureDetector::create("STAR");
    vector<KeyPoint> keypoints_prev;
    detector_h->detect(grayPrev, keypoints_prev);

    vector<KeyPoint> keypoints_prevf;
    detector_f->detect(grayPrev, keypoints_prevf);
    keypoints_prev.insert(keypoints_prev.end(),
                          keypoints_prevf.begin(), keypoints_prevf.end());
    
    

    for (int i = 0; i < keypoints_prev.size(); i++) {
       prevPts.push_back(keypoints_prev[i].pt);
    }

    // do forward backward
    vector<uchar> status;
    vector<uchar> status_back;
    getOpticalFlow(img2, img1, nextPts, prevPts, status);
    getOpticalFlow(img1, img2, backPts, nextPts, status_back);

    std::vector<float> fb_err;
    for (int i = 0; i < prevPts.size(); i++) {
       cv::Point2f v = backPts[i] - prevPts[i];
       fb_err.push_back(sqrt(v.dot(v)));
    }

    float THESHOLD = 25;
    for (int i = 0; i < status.size(); i++) {
       status[i] = (fb_err[i] <= THESHOLD) & status[i];
    }
    
    vector<KeyPoint> keypoints_next;
    for (int i = 0; i < prevPts.size(); i++) {
       Point2f ppt = prevPts[i];
       Point2f npt = nextPts[i];
       double distance = cv::norm(cv::Mat(ppt), cv::Mat(npt));
       if (status[i] && distance > 10) {
          KeyPoint kp;
          kp.pt = nextPts[i];
          kp.size = keypoints_prev[i].size;
          keypoints_next.push_back(kp);
       }
    }



    
    // drawing
    Mat copyimg = img1.clone();
    drawKeypoints(copyimg, keypoints_next, copyimg);
    imshow("key", copyimg);
    
    // compute current keypoints
    vector<KeyPoint>keypoints_cur;
    detector_h->detect(img2, keypoints_cur);

    vector<KeyPoint>keypoints_curf;
    detector_f->detect(img2, keypoints_curf);
    keypoints_cur.insert(keypoints_cur.end(),
                          keypoints_curf.begin(), keypoints_curf.end());
    
    // extract keyponts around good region
    vector<KeyPoint> keypoints_around_region;
    for (int i = 0; i < keypoints_cur.size(); i++) {
       Point2f cur_pt = keypoints_cur[i].pt;
       for (int j = 0; j < keypoints_next.size(); j++) {
          Point2f est_pt = keypoints_next[j].pt;
          double distance = cv::norm(cv::Mat(cur_pt), cv::Mat(est_pt));
          if (distance < 10) {
             keypoints_around_region.push_back(keypoints_cur[i]);
          }
       }
    }

    Mat copyimg2 = img2.clone();
    drawKeypoints(copyimg2, keypoints_cur, copyimg2);
    imshow("key_cur", copyimg2);

    Mat copyimgg = img2.clone();
    drawKeypoints(copyimgg, keypoints_around_region, copyimgg);
    imshow("key_good", copyimgg);
    
    
    Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("ORB");
    Mat descriptor_cur;
    descriptor->compute(img2, keypoints_around_region, descriptor_cur);
    Mat descriptor_prev;
    descriptor->compute(img1, keypoints_prev, descriptor_prev);

    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher =
       cv::DescriptorMatcher::create("BruteForce-Hamming(2)");
    std::vector<std::vector<cv::DMatch> > matchesAll;
    descriptorMatcher->knnMatch(descriptor_cur, descriptor_prev, matchesAll, 2);

    std::vector<DMatch> match1;
    std::vector<DMatch> match2;

    for (int i=0; i < matchesAll.size(); i++) {
       match1.push_back(matchesAll[i][0]);
       match2.push_back(matchesAll[i][1]);
    }
    
    std::vector< DMatch > good_matches;
    for (int i = 0; i < matchesAll.size(); i++) {
       if (match1[i].distance < 0.9 * match2[i].distance) {
          good_matches.push_back(match1[i]);
       }
    }

    
    Mat img_matches1, img_matches2;
    drawMatches(img2, keypoints_around_region, img1,
                keypoints_prev, good_matches, img_matches1);

    /*drawMatches(img2, keypoints_around_region, img1,
                keypoints_prev, match2, img_matches2);
    */
    imshow("matches1", img_matches1);
    // imshow("matches2", img_matches2);
    
    waitKey();
}

void buildImagePyramid(
    const Mat &frame, vector<Mat> &pyramid) {
    
    Mat gray = frame.clone();
    
    Size winSize = Size(5, 5);
    int maxLevel = 5;
    bool withDerivative = true;
    
    buildOpticalFlowPyramid(
       gray, pyramid, winSize, maxLevel, withDerivative,
       BORDER_REFLECT_101, BORDER_CONSTANT, true);
    
}

void getOpticalFlow(
    const Mat &frame, const Mat &prevFrame,
    vector<Point2f> &nextPts, vector<Point2f> &prevPts,
    vector<uchar> &status) {
    Mat gray, grayPrev;
    cvtColor(prevFrame, grayPrev, CV_BGR2GRAY);
    cvtColor(frame, gray, CV_BGR2GRAY);
    vector<Mat> curPyramid;
    vector<Mat> prevPyramid;
    buildImagePyramid(frame, curPyramid);
    buildImagePyramid(prevFrame, prevPyramid);
    
    // vector<uchar> status;
    vector<float> err;
    nextPts.clear();
    status.clear();
    nextPts.resize(prevPts.size());
    status.resize(prevPts.size());
    err.resize(prevPts.size());
        
    Size winSize = Size(5, 5);
    int maxLevel = 3;
    TermCriteria criteria = TermCriteria(
       TermCriteria::COUNT+TermCriteria::EPS, 30, 0.001);
    int flags = 1;
        
    calcOpticalFlowPyrLK(
       prevPyramid, curPyramid, prevPts, nextPts, status,
       err, winSize, maxLevel, criteria, flags);
    // Mat iFrame = frame.clone();
    Mat iFrame = prevFrame.clone();
    
    drawFlowField(iFrame, nextPts, prevPts);
}






void drawFlowField(Mat &frame, vector<Point2f> &nextPts, vector<Point2f> &prevPts){

    vector<Point2f> good_match;
    vector<Point2f> good_match_prev;
    for (int i = 0 ; i < nextPts.size(); i++) {
        double angle = atan2((double)(prevPts[i].y - nextPts[i].y),(double)(prevPts[i].x - nextPts[i].x));
        double lenght = sqrt((double)pow(prevPts[i].y - nextPts[i].y, 2) + (double)pow((double)(prevPts[i].x - nextPts[i].x), 2));
        // std::cout << lenght << std::endl;
        if (lenght > 0) {
           good_match.push_back(nextPts[i]);
           good_match_prev.push_back(prevPts[i]);
           
           Point iBig = Point(prevPts[i].x - 1 * lenght * cos(angle),
                              prevPts[i].y - 1 * lenght * sin(angle));
           line(frame, prevPts[i], iBig, Scalar(255, 0, 255));
        
           //!Draw the arrow on the line
           line(frame, Point2f(iBig.x + 1 * cos(
                                  angle + CV_PI/4), iBig.y + 1 * sin(
                                     angle + CV_PI/4)), iBig, Scalar(255, 0, 255));
           line(frame, Point2f(iBig.x + 1 * cos(
                                  angle - CV_PI/4), iBig.y + 1 * sin(angle - CV_PI/4)),
                iBig, Scalar(255, 0, 255));
        }
    }
    // nextPts.clear();
    // prevPts.clear();
    // prevPts = good_match_prev;
    // nextPts = good_match;
    imshow("flow", frame);
}

void computeAffineTransformation(const vector<Point2f> &nextPts, const vector<Point2f> &prevPts){
 
    cout << nextPts.size() << "  " << prevPts.size() << endl;
    Mat affine = estimateRigidTransform(nextPts, prevPts, true);
    
    cout << endl << affine << endl << endl;
}

string float2string(float c_frame){
    
    std::string frame_num;
    std::stringstream out;
    out << c_frame;
    frame_num = out.str();
    
    return frame_num;
}
