
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <string>

#include "lbp.hpp"
#include "histogram.hpp"

using namespace cv;

int main(int argc, const char *argv[]) {
	int deviceId = 0;
  
	// initial values
    int radius = 3;
    int neighbors = 8;
    Mat frame;
    frame = imread("/home/krishneel/Desktop/image.jpg");
    // frame = imread("/home/krishneel/Desktop/drill_00078.jpg");
    // resize(frame, frame, Size(640, 480));
    
    Mat dst;  // image after preprocessing
    Mat lbp;  // lbp image

    // just to switch between possible lbp operators
    vector<string> lbp_names;
    lbp_names.push_back("Extended LBP");  // 0
    lbp_names.push_back("Fixed Sampling LBP");  // 1
    lbp_names.push_back("Variance-based LBP");  // 2
    int lbp_operator=atoll(argv[1]);;
    
    //  while(running) {

    cvtColor(frame, dst, CV_BGR2GRAY);
    GaussianBlur(dst, dst, Size(7, 7), 5, 3, BORDER_CONSTANT);

    	// comment the following lines for original size
    // resize(frame, frame, Size(), 0.5, 0.5);
    // resize(dst, dst, Size(), 0.5, 0.5);
    	
    switch (lbp_operator) {
       case 0:
          lbp::ELBP(dst, lbp, radius, neighbors);  // use the extended operator
          break;
       case 1:
          lbp::OLBP(dst, lbp);  // use the original operator
          break;
       case 2:
          lbp::VARLBP(dst, lbp, radius, neighbors);
          break;
    }
    	// now to show the patterns a normalization is necessary
    	// a simple min-max norm will do the job...
    normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);

    imshow("original", frame);
    imshow("lbp", lbp);
    waitKey(0);
    
    char key = (char) waitKey(20);
    
    
    	// to make it a bit interactive, you can increase and decrease the parameters
    switch (key) {
         case 'q': case 'Q':
            break;
       case 'r':
          radius -= 1;
          radius = std::max(radius,1);
          cout << "radius=" << radius << endl;
          break;
          // upper case r increases the radius (there's no real upper bound)
       case 'R':
          radius+=1;
          radius = std::min(radius,32);
          cout << "radius=" << radius << endl;
          break;
          // lower case p decreases the number of sampling points (min 1)
       case 'p':
          neighbors -= 1;
          neighbors = std::max(neighbors,1);
          cout << "sampling points=" << neighbors << endl;
          break;
          // upper case p increases the number of sampling points (max 31)
       case 'P':
          neighbors+=1;
          neighbors = std::min(neighbors,31);
          cout << "sampling points=" << neighbors << endl;
          break;
          // switch between operators
       case 'o': case 'O':
          lbp_operator = (lbp_operator + 1) % 3;
          cout << "Switched to operator " << lbp_names[lbp_operator] << endl;
          break;
       case 's': case 'S':
          imwrite("original.jpg", frame);
          imwrite("lbp.jpg", lbp);
          cout << "Screenshot (operator="
               << lbp_names[lbp_operator] << ",radius="
               << radius <<",points=" << neighbors << ")" << endl;
          break;
       default:
          break;
        //}

        cv::imshow("lbp", dst);
        cv::waitKey(0);
    }
    	return 0; // success
}
