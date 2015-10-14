
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
       const char* v_name;

       VertexProperty(
          int i = -1, const char* name = "default") :
          v_index(i), v_name(name) {}
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

    const char *name[7];
   
    void generateRAG();
    void regionMergingGraph();
    void printGraph(const Graph &);
    void updateRAG(
      Graph &, const std::vector<VertexDescriptor> &, bool = false);
};

RegionAdjacencyGraph::RegionAdjacencyGraph() {

    name[0] = "Jeanie";
    name[1] = "Debbie";
    name[2] = "Rick";
    name[3] = "John";
    name[4] = "Amanda";
    name[5] = "Margaret";
    name[6] = "Benjamin";
}


void RegionAdjacencyGraph::generateRAG() {
   
     VertexDescriptor Jeanie = add_vertex(
        VertexProperty(0, name[0]), this->graph);
     VertexDescriptor Debbie = add_vertex(
        VertexProperty(1, name[1]), this->graph);
     VertexDescriptor Rick = add_vertex(
        VertexProperty(2, name[2]), this->graph);
     VertexDescriptor John = add_vertex(
        VertexProperty(3, name[3]), this->graph);
     VertexDescriptor Amanda = add_vertex(
        VertexProperty(4, name[4]), this->graph);
     VertexDescriptor Margaret = add_vertex(
        VertexProperty(5, name[5]), this->graph);
     VertexDescriptor Benjamin = add_vertex(
        VertexProperty(6, name[6]), this->graph);
     
     add_edge(Jeanie, Debbie, EdgeProperty(0.5f), this->graph);
     add_edge(Jeanie, Rick, EdgeProperty(0.2f), this->graph);
     add_edge(Jeanie, John, EdgeProperty(0.1f), this->graph);
     add_edge(Debbie, Amanda, EdgeProperty(0.3f), this->graph);
     add_edge(Rick, Margaret, EdgeProperty(0.4f), this->graph);
     add_edge(John, Benjamin, EdgeProperty(-0.7f), this->graph);
}

/**
 * Create and process the graph
 */

void RegionAdjacencyGraph::regionMergingGraph() {


     IndexMap index_map = get(boost::vertex_index, graph);
     EdgePropertyAccess edge_weights = get(boost::edge_weight, graph);
     VertexIterator i, end;
     std::vector<VertexDescriptor> to_remove;

     for (tie(i, end) = vertices(graph); i != end; i++) {
         std::cout << name[get(index_map, *i)];
         AdjacencyIterator ai, a_end;
         tie(ai, a_end) = adjacent_vertices(*i, graph);

         if (ai == a_end) {
             std::cout << " has no children\n\n";
         } else {
             std::cout << " is the parent of ";
         }
         
         std::vector<VertexDescriptor> to_clear;
         for (; ai != a_end; ++ai) {
           bool found = false;
           EdgeDescriptor e_descriptor;
           tie(e_descriptor, found) = edge(*i, *ai, graph);
           if (found) {
              EdgeValue edge_val = boost::get(
                 boost::edge_weight, graph, e_descriptor);
              float weights_ = edge_val;
              if (weights_ > THRESHOLD) {
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
                           *i, *aI, EdgeProperty(static_cast<float>(e_val)), graph);

                        std::cout << "Is Added: " << located << "\t" << ed
                                  << "\tEdge weight: " << e_val << std::endl;
                        // std::cout << " DEBUG:  " << *i  << "  "
                        //           << *ai  << "  " 
                        //           << *aI << " \n" ;
                    }
                    std::cout << " -- OUT DEBUG:  " << *i  << "  "
                              << *ai  << "  " << *aI << " \n";
                 }
              }
           }
         }
         updateRAG(graph, to_clear);
         to_clear.clear();
     }
     updateRAG(graph, to_remove, true);
     printGraph(graph);
     std::cout << "\nGraph Size: " << num_vertices(graph) <<
     std::endl;
}


/**
 * Print the tree
 */
void RegionAdjacencyGraph::printGraph(const Graph &graph) {
    VertexIterator i, end;
    for (tie(i, end) = vertices(graph); i != end; ++i) {
        std::cout  << "- Graph Property: "
                   << graph[*i].v_index << "\t"
                   << graph[*i].v_name   << std::endl;
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
        std::cout << graph[it].v_index << "\t" << graph[it].v_name << std::endl;
        clear_vertex(it, graph);
        if (is_rem) {
            remove_vertex(it, graph);
        }
    }
}



/**
 * 
 */
int main(int argc, const char *argv[]) {


    RegionAdjacencyGraph rag;
    rag.generateRAG();
    rag.regionMergingGraph();
    
    exit(1);
   
   
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
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
    
    
    
    imshow("image", image);
    waitKey(0);

    return EXIT_SUCCESS;
}
