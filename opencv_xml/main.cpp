
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>


void writeFile(cv::FileStorage &fs) {
   // std::string classifername = "cv::SVM";
    std::string classiferpath = "~/.ros/svm.xml";
    
    fs <<  "TrainerInfo" << "{";
    fs << "trainer_type" << "cv::SVM";
    fs << "trainer_path" << classiferpath;
    fs << "}";

    fs <<  "FeatureInfo" << "{";
    fs << "HOG" << 1;
    fs << "LBP" << 1;
    fs << "SIFT" << 0;
    fs << "SURF" << 0;
    fs << "COLOR_HISTOGRAM" << 0;
    fs << "}";
    
    fs <<  "SlidingWindowInfo" << "{";
    fs << "window_x" << 32;
    fs << "window_y" << 64;
    fs << "}";
    
    fs << "TrainingDatasetDirectory" << "{";
    fs << "object_dataset" << "~/.ros/";
    fs << "nonobject_dataset" << "~/.ros/";
    fs << "}";
}

void readFile(cv::FileStorage &fs) {
    if (!fs.isOpened()) {
       std::cout << "Empty File..." << std::endl;
       return;
    }
    cv::FileNode n = fs["TrainerInfo"];
    std::string ttype = n["trainer_type"];
    std::string tpath = n["trainer_path"];
    std::cout << ttype << std::endl;
    std::cout << tpath << std::endl;

    n = fs["FeatureInfo"];
    int hog = n["HOG"];
    int lbp = n["LBP"];
    std::cout << hog << std::endl;
    std::cout << lbp << std::endl;
    
    
    n = fs["SlidingWindowInfo"];
    int swindow_x = static_cast<int>(n["window_x"]);
    int swindow_y = static_cast<int>(n["window_y"]);

    n = fs["TrainingDatasetDirectory"];
    std::string pfile = n["object_dataset"];
    std::string nfile = n["nonobject_dataset"];

    std::cout << swindow_x << " " << swindow_y << std::endl;
    std::cout << pfile << " \n" << nfile << std::endl;
}

int main(int argc, char *argv[]) {

    std::string filename = "trainer_manifest.xml";
    cv::FileStorage fs = cv::FileStorage(filename, cv::FileStorage::WRITE);
    writeFile(fs);
    fs.release();

    std::cout << "READIND \n" << std::endl;

    cv::FileStorage rfs;
    rfs.open(filename, cv::FileStorage::READ);
    readFile(rfs);
    
    
    return 0;
}

