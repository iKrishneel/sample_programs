
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
class RegionAdjacencyGraph {

#define THRESHOLD (0.0)

 private:
    struct VertexProperty {
       int v_index;
       cv::Point2f v_center;
       
       VertexProperty(
          int i = -1, cv::Point2f center = cv::Point2f(-1, -1)) :
          v_index(i), v_center(center) {}
    };
    typedef boost::property<boost::edge_weight_t, float> EdgeProperty;
    typedef typename boost::adjacency_list<boost::vecS,
                                           boost::vecS,
                                           boost::undirectedS,
                                           VertexProperty,
                                           EdgeProperty> Graph;
    typedef typename boost::graph_traits<
       Graph>::adjacency_iterator AdjacencyIterator;
    typedef typename boost::property_map<
      Graph, boost::vertex_index_t>::type IndexMap;
    typedef typename boost::graph_traits<
       Graph>::edge_descriptor EdgeDescriptor;
    typedef typename boost::property_map<
       Graph, boost::edge_weight_t>::type EdgePropertyAccess;
    typedef typename boost::property_traits<boost::property_map<
      Graph, boost::edge_weight_t>::const_type>::value_type EdgeValue;
    typedef typename boost::graph_traits<
       Graph>::vertex_iterator VertexIterator;
    typedef typename boost::graph_traits<
       Graph>::vertex_descriptor VertexDescriptor;
   
    Graph graph;
   
   
 public:
    RegionAdjacencyGraph();
    void generateRAG(
       const vector<cv::Mat> &, const cv::Mat &, const cv::Mat &);
    void regionMergingGraph(
       const vector<cv::Mat> &, const cv::Mat &);
    void printGraph(const Graph &);
    void updateRAG(
      Graph &, const std::vector<VertexDescriptor> &, bool = false);
    void computeHistogram(
      const cv::Mat &, cv::Mat &, bool = true);
    void concatenateRegion(
      const cv::Mat &, const cv::Point2i &, cv::Mat &);
};

/**
 * constructor
 */
RegionAdjacencyGraph::RegionAdjacencyGraph() {
   
}


void RegionAdjacencyGraph::generateRAG(
    const vector<cv::Mat> &patches,
    const cv::Mat &neigbours,
    const cv::Mat &centroids) {
    if (neigbours.empty() || centroids.empty()) {
       std::cout << "Error" << std::endl;
       return;
    }
    if (neigbours.rows == centroids.rows) {
       std::vector<VertexDescriptor> vertex_descriptor;
       int icounter = 0;
       for (int j = 0; j < neigbours.rows; j++) {
          float c_x = static_cast<float>(centroids.at<float>(j, 0));
          float c_y = static_cast<float>(centroids.at<float>(j, 1));
          VertexDescriptor v_des = add_vertex(
             VertexProperty(j, cv::Point2f(c_x, c_y)), this->graph);
          vertex_descriptor.push_back(v_des);
       }
       for (int j = 0; j < neigbours.rows; j++) {
          VertexDescriptor r_vd = vertex_descriptor[j];
          cv::Mat pHist;
          this->computeHistogram(patches[j], pHist);
          for (int i = 1; i < neigbours.cols; i++) {
             int n_index = neigbours.at<int>(j, i);
             VertexDescriptor vd = vertex_descriptor[n_index];
             cv::Mat nHist;
             this->computeHistogram(patches[n_index], nHist);
             float distance = static_cast<float>(
                compareHist(pHist, nHist, CV_COMP_BHATTACHARYYA));
             if (r_vd != vd) {
                bool found = false;
                EdgeDescriptor e_descriptor;
                tie(e_descriptor, found) = edge(r_vd, vd, graph);
                if (!found) {
                   boost::add_edge(
                      r_vd, vd, EdgeProperty(distance), this->graph);
                }
             }
          }
       }
    }
}

/**
 * Create and process the graph
 */
void RegionAdjacencyGraph::regionMergingGraph(
    const vector<cv::Mat> &patches,
    const Mat &centroids) {
   
     IndexMap index_map = get(boost::vertex_index, graph);
     EdgePropertyAccess edge_weights = get(boost::edge_weight, graph);
     VertexIterator i, end;
     std::vector<VertexDescriptor> to_remove;

     cv::Mat segmented_image = Mat::zeros(480, 640, CV_8UC3);
     int icounter = 0;
     
     for (tie(i, end) = vertices(graph); i != end; i++) {
         AdjacencyIterator ai, a_end;
         tie(ai, a_end) = adjacent_vertices(*i, graph);
         std::vector<VertexDescriptor> to_clear;

         // std::cout << *i << "\t";
         
         for (; ai != a_end; ++ai) {
            // std::cout << *ai << "  ";
            
            
           bool found = false;
           EdgeDescriptor e_descriptor;
           tie(e_descriptor, found) = edge(*i, *ai, graph);
           if (found) {
              EdgeValue edge_val = boost::get(
                 boost::edge_weight, graph, e_descriptor);
              float weights_ = edge_val;

              std::cout << "Vertex: " << *ai
                        << "\tWeight G(V,E): " << weights_ << std::endl;
              
              if (weights_ < 0.5) {
                 // merge the regions

                 int x_ = centroids.at<float>(static_cast<int>(*i), 0);
                 int y_ = centroids.at<float>(static_cast<int>(*i), 1);
                 cv::Point2i centr(x_, y_);
                 this->concatenateRegion(
                    patches[*i], centr, segmented_image);
                
                 x_ = centroids.at<float>(static_cast<int>(*ai), 0);
                 y_ = centroids.at<float>(static_cast<int>(*ai), 1);
                 centr = cv::Point2i(x_, y_);
                 this->concatenateRegion(
                    patches[*ai], centr, segmented_image);

                 to_remove.push_back(*ai);
                 to_clear.push_back(*ai);
                 AdjacencyIterator aI, aEnd;
                 tie(aI, aEnd) = adjacent_vertices(*ai, graph);
                 for (; aI != aEnd; aI++) {
                    EdgeDescriptor ed;
                    bool located;
                    tie(ed, located) = edge(*aI, *ai, graph);
                    if (located && *aI != *i) {
                        EdgeValue e_val = boost::get(
                            boost::edge_weight, graph, ed);
                        tie(ed, located) = add_edge(
                           *i, *aI, EdgeProperty(
                              static_cast<float>(e_val)), graph);
/*
                        std::cout << "--- IsAdded:  " << located << "\t"
                                  << ed << "\t Edge Weight: " << e_val
                                  << std::endl;
                         std::cout << "\n DEBUG:  " << *i  << "  "
                              << *ai  << "  "
                              << *aI << " \n";
*/
                    }
                 }
              }
           }
         }
         updateRAG(graph, to_clear);
         to_clear.clear();

         if (icounter++ > 150) {
            break;
         }

         // } std::cout << std::endl;
         
         
     }
     updateRAG(graph, to_remove, true);
     // printGraph(graph);
     std::cout << "\nGraph Size: " << num_vertices(graph) <<
     std::endl;

     if (segmented_image.data) {
       imshow("RAG", segmented_image);
    }
}


/**
 * Print the tree
 */
void RegionAdjacencyGraph::printGraph(
    const Graph &graph) {
    VertexIterator i, end;
    for (tie(i, end) = vertices(graph); i != end; ++i) {
        std::cout  << "- Graph Property: "
                   << graph[*i].v_index << "\t"
                   << graph[*i].v_center   << std::endl;
    }
}

/**
 * updating the graph
 */
void RegionAdjacencyGraph::updateRAG(
    Graph &graph,
    const std::vector<VertexDescriptor> &to_remove,
    bool is_rem) {
    for (int i = to_remove.size() - 1; i >= 0; i--) {
        VertexDescriptor it = to_remove.at(i);
        std::cout << graph[it].v_index << "\t"
                  << graph[it].v_center << std::endl;
        clear_vertex(it, graph);
        if (is_rem) {
            remove_vertex(it, graph);
        }
    }
}

/**
 * function to merge the 2 given node and the corresponding regions
 */
void RegionAdjacencyGraph::concatenateRegion(
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

/**
 * compute the histogram
 */
void RegionAdjacencyGraph::computeHistogram(
    const cv::Mat &src,
    cv::Mat &hist,
    bool isNormalized) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    int hBin = 30;
    int sBin = 30;
    int histSize[] = {hBin, sBin};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    cv::calcHist(
       &hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    if (isNormalized) {
        cv::normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
    }
}

/**
 * 
 */
int main(int argc, const char *argv[]) {
   
    cv::Mat image = cv::imread("room2.jpg");
    if (image.empty()) {
       std::cout << "NO IMAGE FOUND!!" << std::endl;
       return EXIT_FAILURE;
    }
    cv::resize(image, image, cv::Size(640, 480));
    
    int width = 40;
    int height = 30;
    
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

    Mat neigbour_index;
    Mat dist;
    kdtree.knnSearch(
       centroid, neigbour_index, dist, 4, cv::flann::SearchParams(64));
    
    RegionAdjacencyGraph rag;
    rag.generateRAG(patches, neigbour_index, centroid);
    rag.regionMergingGraph(patches, centroid);

    std::cout << neigbour_index << std::endl;
    
    // imshow("image", image);
    waitKey(0);

    return EXIT_SUCCESS;
}
