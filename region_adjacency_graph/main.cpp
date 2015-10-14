
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

#define THRESHOLD (0.65)
   
 private:
    struct VertexProperty {
       int v_index;
       cv::Point2f v_center;
       int v_label;
       
       VertexProperty(
          int i = -1,
          cv::Point2f center = cv::Point2f(-1, -1),
          int label = -1) :
          v_index(i), v_center(center), v_label(label) {}
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
    void computeHistogram(
      const cv::Mat &, cv::Mat &, bool = true);
    void concatenateRegion(
      const cv::Mat &, const cv::Point2i &, cv::Mat &);
    void getImageGridLabels(std::vector<int> &);
   
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
    if (num_vertices(this->graph) == 0) {
       std::cout << "Empty Graph..." << std::endl;
       return;
    }
    IndexMap index_map = get(boost::vertex_index, this->graph);
    EdgePropertyAccess edge_weights = get(boost::edge_weight, this->graph);
    VertexIterator i, end;
    int label = -1;
    for (tie(i, end) = vertices(this->graph); i != end; i++) {
        if (this->graph[*i].v_label == -1) {
           graph[*i].v_label = ++label;
        }
        AdjacencyIterator ai, a_end;
        tie(ai, a_end) = adjacent_vertices(*i, this->graph);
        for (; ai != a_end; ++ai) {
           bool found = false;
           EdgeDescriptor e_descriptor;
           tie(e_descriptor, found) = edge(*i, *ai, this->graph);
           if (found) {
              EdgeValue edge_val = boost::get(
                 boost::edge_weight, this->graph, e_descriptor);
              float weights_ = edge_val;
              if (weights_ > THRESHOLD) {
                 remove_edge(e_descriptor, this->graph);
              } else {
                if (this->graph[*ai].v_label == -1) {
                   this->graph[*ai].v_label = this->graph[*i].v_label;
                }
              }
           }
        }
     }
     this->printGraph(this->graph);
     std::cout << "\nPRINT INFO. \n --Graph Size: " << num_vertices(graph) <<
        std::endl << "--Total Label: " << label << "\n\n";
}


/**
 * Print the tree
 */
void RegionAdjacencyGraph::printGraph(
    const Graph &_graph) {
    VertexIterator i, end;
    int icount = 0;
   
    for (tie(i, end) = vertices(_graph); i != end; ++i) {

       std::cout << _graph[*i].v_label << "   ";
       
       /*
       AdjacencyIterator ai, a_end;
       tie(ai, a_end) = adjacent_vertices(*i, _graph);
       icount++;
       for (; ai != a_end; ++ai) {
          bool found = false;
          EdgeDescriptor e_descriptor;
          tie(e_descriptor, found) = edge(*i, *ai, _graph);
          if (found) {
             VertexDescriptor s = boost::source(e_descriptor, graph);
             VertexDescriptor t = boost::target(e_descriptor, graph);
             std::cout << graph[s].v_label << "," << graph[t].v_label << "\t ";
             
             // std::cout << e_descriptor << "  "
             //           << s <<  "  "
             //           << t << "\t";
          }
       }std::cout << "\n" << std::endl;
       */
       // std::cout << *i << "\t" << _graph[*i].v_label << std::endl;
    }
    std::cout << "Final Count: " << icount << std::endl;
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
    int hBin = 10;
    int sBin = 10;
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
void RegionAdjacencyGraph::getImageGridLabels(std::vector<int> &labelMD) {
       VertexIterator i, end;
    for (tie(i, end) = vertices(this->graph); i != end; ++i) {
       labelMD.push_back(static_cast<int>(this->graph[*i].v_label));
    }
}


// -----------------------------
void makeLabelImage(
    const std::vector<int> &labelMD,
    std::vector<Rect_<float> > region, cv::Size _sz) {
    if (labelMD.size() != region.size()) {
       std::cout << "Error Not Same Size" << std::endl;
       return;
    }
    cv::RNG rng(12345);
    int gen_size = labelMD.size() + 1;
    cv::Scalar color[gen_size];
    for (int i = 0; i < gen_size; i++) {
       color[i] = cv::Scalar(
          rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    cv::Mat image = cv::Mat(_sz, CV_8UC3);
    for (int i = 0; i < labelMD.size(); i++) {
       int label = labelMD.at(i);
       cv::Rect_<float> rect = region.at(i);
       cv::rectangle(image, rect, color[label], -1);
    }
    imshow("image", image);
}

/**
 * 
 */
int main(int argc, const char *argv[]) {

    srand(time(NULL));
    cv::Mat image = cv::imread("room.jpg");
    if (image.empty()) {
       std::cout << "NO IMAGE FOUND!!" << std::endl;
       return EXIT_FAILURE;
    }
    cv::resize(image, image, cv::Size(640, 480));
    
    int width = 20;
    int height = 15;
    
    vector<Mat> patches;
    std::vector<cv::Rect_<float> > region;
    
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
             region.push_back(_rect);
             Point2f _center = Point2f(_rect.x + _rect.width/2,
                                       _rect.y + _rect.height/2);
             // centroid.push_back(_center);
             centroid.at<float>(y, 0) = _center.x;
             centroid.at<float>(y++, 1) = _center.y;
          }
       }
    }
    cv::flann::KDTreeIndexParams indexParams(2);
    cv::flann::Index kdtree(centroid, indexParams);

    Mat neigbour_index;
    Mat dist;
    kdtree.knnSearch(
       centroid, neigbour_index, dist, 8, cv::flann::SearchParams(64));
    
    RegionAdjacencyGraph rag;
    rag.generateRAG(patches, neigbour_index, centroid);
    rag.regionMergingGraph(patches, centroid);
    std::vector<int> labelMD;
    rag.getImageGridLabels(labelMD);
    
    
    makeLabelImage(labelMD, region, image.size());
    imshow("original", image);
    waitKey(0);

    return EXIT_SUCCESS;
}
