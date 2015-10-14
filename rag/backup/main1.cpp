
#include <boost/graph/adjacency_list.hpp>
#include <boost/tuple/tuple.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;


class RAG {
   
#define THRESHOLD (0.5)
   
 private:
    typedef boost::property<boost::vertex_index_t, int> VertexProperty;
    typedef boost::property<boost::edge_weight_t, float> EdgeProperty;

    //  the graph structure
    boost::adjacency_list <boost::vecS,
                           boost::vecS,
                           boost::undirectedS,
                           VertexProperty,
                           EdgeProperty> graph;
    //  prototype for edge descriptor
    typedef typename boost::graph_traits<
       boost::adjacency_list<boost::vecS,
                             boost::vecS,
                             boost::undirectedS,
                             VertexProperty,
                             EdgeProperty>
       >::edge_descriptor EdgeDescriptor;
    // prototype to handle the edge values
    typedef typename boost::property_traits<boost::property_map<
       boost::adjacency_list<boost::vecS,
                             boost::vecS,
                             boost::undirectedS,
                             VertexProperty,
                             EdgeProperty>,
       boost::edge_weight_t>::const_type>::value_type EdgeValue;
    // prototype for vertex descriptor
    boost::graph_traits<
       boost::adjacency_list<boost::vecS,
                             boost::vecS,
                             boost::undirectedS,
                             VertexProperty,
                             EdgeProperty>
       >::vertex_descriptor VertexDescriptor;
    // prototype for vertex iteration
    typedef typename boost::graph_traits<
       boost::adjacency_list<boost::vecS,
                             boost::vecS,
                             boost::undirectedS,
                             VertexProperty,
                             EdgeProperty>
       >::vertex_iterator VertexIterator;
    // prototype to accessing adjacent vertices
    typedef typename boost::graph_traits<
       boost::adjacency_list<boost::vecS,
                             boost::vecS,
                             boost::undirectedS,
                             VertexProperty,
                             EdgeProperty>
       >::adjacency_iterator AdjacencyIterator;

   
    void concatenateRegion(
       const cv::Mat &,
       const cv::Point2i &,
       cv::Mat &);

    void computeHistogram(const Mat &, Mat &, bool = true);
   
   
   
 public:
   // RAG();

   
    void generateRAG(
       const vector<cv::Mat> &,
       const Mat &,
       const Mat &);
   
    bool mergeRegionRAG(
       const std::vector<cv::Mat> &,
       const Mat &);
};

void RAG::generateRAG(
    const vector<cv::Mat> &patches,
    const Mat &neigbours,
    const Mat &centroids) {
    if (neigbours.empty() || centroids.empty()) {
       std::cout << "Error" << std::endl;
       return;
    }
    if (neigbours.rows == centroids.rows) {
       int icounter = 0;
       for (int j = 0; j < neigbours.rows; j++) {
          int c_index = j;  // static_cast<int>(neigbours.at<uchar>(j, 0));
          add_vertex(c_index, this->graph);

          Mat pHist;
          this->computeHistogram(patches[c_index], pHist);
          
          for (int i = 0; i < neigbours.cols; i++) {
             int n_index = static_cast<int>(neigbours.at<uchar>(j, i));

             Mat nHist;
             this->computeHistogram(patches[n_index], nHist);
             
             float distance = 0.0f;  // get weight function
             distance = static_cast<float>(
                compareHist(pHist, nHist, CV_COMP_BHATTACHARYYA));
             add_edge(c_index, n_index, EdgeProperty(distance), this->graph);
          }
       }
    }
    std::cout << "Graph Size: " << num_vertices(this->graph) << std::endl;
}


bool RAG::mergeRegionRAG(
    const vector<cv::Mat> &patches,
    const Mat &centroids) {
    if (num_vertices(this->graph) == 0 || patches.empty()) {
       std::cout << "ERROR: Empty Graph" << std::endl;
       return false;
    }
    boost::property_map<boost::adjacency_list<boost::vecS,
                                               boost::vecS,
                                               boost::undirectedS,
                                               VertexProperty,
                                               EdgeProperty>,
                         boost::vertex_index_t >::type
     vertex_index_map = get(boost::vertex_index, this->graph);
     VertexIterator vIter_begin, vIter_end;
     EdgeDescriptor e_descriptor;
     AdjacencyIterator aIter_begin, aIter_end;
     
     cv::Mat segmented_image = Mat::zeros(480, 640, CV_8UC3);
     
    for (tie(vIter_begin, vIter_end) = vertices(this->graph);
        vIter_begin != vIter_end; ++vIter_begin) {
       tie(aIter_begin, aIter_end) = adjacent_vertices(
          *vIter_begin, this->graph);
       for (; aIter_begin != aIter_end; ++aIter_begin) {
          bool found;
          tie(e_descriptor, found) = edge(
             *vIter_begin, *aIter_begin, this->graph);
          if (found) {
             EdgeValue edge_val = boost::get(
                boost::edge_weight, this->graph, e_descriptor);
             float e_weights = edge_val;
             if (e_weights > THRESHOLD) {
                // merge the regions
                int x_ = centroids.at<float>(static_cast<int>(*vIter_begin), 0);
                int y_ = centroids.at<float>(static_cast<int>(*vIter_begin), 1);
                cv::Point2i centr(x_, y_);
                this->concatenateRegion(
                   patches[*vIter_begin], centr, segmented_image);
                
                x_ = centroids.at<float>(static_cast<int>(*aIter_begin), 0);
                y_ = centroids.at<float>(static_cast<int>(*aIter_begin), 1);
                centr = cv::Point2i(x_, y_);
                this->concatenateRegion(
                   patches[*aIter_begin], centr, segmented_image);
                
                
                /* Direct all the edges to the new vertex and than
                 * delete the other vertex
                 */
                AdjacencyIterator aI;
                AdjacencyIterator aEnd;
                tie(aI, aEnd) = adjacent_vertices(*vIter_begin, this->graph);
                for (; aI != aEnd; aI++) {
                   EdgeDescriptor ed;
                   bool located = false;
                   tie(ed, located) = edge(*aI, *aIter_begin, this->graph);
                   if (located && *aI != *vIter_begin) {
                      EdgeValue ev = boost::get(
                         boost::edge_weight, this->graph, ed);
                      /*add_edge(*aI,
                               *vIter_begin,
                               EdgeProperty(static_cast<float>(ev)),
                               this->graph);*/
                      std::cout << "Vertex: " << *aI << " "
                                << get(vertex_index_map, *aEnd) << "\n";
                   }
                }
                
                // clear_vertex(*aIter_begin, this->graph);
                // remove_vertex(*aIter_begin, this->graph);
                
                std::cout << "Common: " << e_descriptor << std::endl;
                
                
                // clear_vertex(*aIter_begin, this->graph);
                // remove_vertex(*aIter_begin, this->graph);
                
             }
          }
       }
    }
    if (segmented_image.data) {
       imshow("RAG", segmented_image);
    }
}



/**
 * function to merge the 2 given node and the corresponding regions
 */
void RAG::concatenateRegion(
    const cv::Mat &n_region,
    const cv::Point2i &centroids,
    cv::Mat &out_img) {
    int x_off = centroids.x - n_region.cols/2;
    int y_off = centroids.y - n_region.rows/2;
    for (int j = 0; j < n_region.rows; j++) {
       for (int i = 0; i < n_region.cols; i++) {
          out_img.at<cv::Vec3b>(y_off + j, x_off + i) =
             n_region.at<cv::Vec3b>(j, i);
       }
    }
}


void RAG::computeHistogram(const Mat &src, Mat &hist, bool isNormalized) {
    Mat hsv;
    cvtColor(src, hsv, CV_BGR2HSV);

    int hBin = 16;
    int sBin = 16;
    int histSize[] = {hBin, sBin};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    
    const float* ranges[] = {h_ranges, s_ranges};
    
    int channels[] = {0, 1};
    calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    
    if (isNormalized) {
        normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
    }
}


int main(int argc, const char *argv[]) {

    // boostSampleGraph();
    // exit(-1);
   
   
    cv::Mat image = cv::Mat(480, 640, CV_8UC3, Scalar(255, 255, 255));
    Rect rect = Rect(64, 96, 256, 96);
    rectangle(image, rect, Scalar(0, 255, 0), -1);

    
    int width = 64;
    int height = 48;
    
    vector<Mat> patches;
    int _num_element = (image.rows/height) * (image.cols/width);
    Mat centroid = Mat(_num_element, 2, CV_32F);

    int y = 0;
    for (int j = 0; j < image.rows; j += height) {
       for (int i = 0; i < image.cols; i += width) {
          Rect_<float> _rect = Rect_<float>(i, j, width, height);
          if (_rect.x + _rect.width <= image.cols &&
              _rect.y + _rect.height <= image.rows) {
             Mat roi = image(_rect);
             patches.push_back(roi);
             Point2f _center = Point2f(_rect.x + _rect.width/2,
                                       _rect.y + _rect.height/2);
             // centroid.push_back(_center);
             centroid.at<float>(y, 0) = _center.x;
             centroid.at<float>(y++, 1) = _center.y;
          }
       }
    }
    
    cv::flann::KDTreeIndexParams indexParams(5);
    cv::flann::Index kdtree(centroid, indexParams);

    Mat index;
    Mat dist;
    kdtree.knnSearch(centroid, index, dist, 4, cv::flann::SearchParams(64));

    RAG rag;
    rag.generateRAG(patches, index, centroid);
    rag.mergeRegionRAG(patches, centroid);
    
    
    imshow("image", image);
    waitKey(0);

    return EXIT_SUCCESS;
}
