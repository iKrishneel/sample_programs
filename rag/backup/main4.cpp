
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


enum family {
    Jeanie, Debbie, Rick, John, Amanda, Margaret, Benjamin, N };
const char *name[] = {
    "Jeanie", "Debbie", "Rick", "John", "Amanda", "Margaret", "Benjamin" };


typedef boost::property<boost::vertex_index_t, int> vertex_property;
typedef boost::property<boost::edge_weight_t, float> edge_property;
typedef typename boost::adjacency_list <boost::vecS,
                                        boost::vecS,
                                        boost::undirectedS,
                                        vertex_property,
                                        edge_property> Graph;
typedef typename boost::graph_traits<
     Graph>::adjacency_iterator AdjacencyIterator;
typedef typename boost::property_map<
     Graph, boost::vertex_index_t >::type IndexMap;
typedef typename boost::graph_traits<
     Graph>::edge_descriptor EdgeDescriptor;
typedef typename boost::property_map<
     Graph, boost::edge_weight_t>::type EdgePropertyAccess;
typedef typename boost::property_traits<boost::property_map<
     Graph, boost::edge_weight_t>::const_type>::value_type EdgeValue;
typedef typename boost::graph_traits<
    Graph>::vertex_iterator VectorIterator;

/**
 * updating the graph
 */
bool updateRAG(const Graph &graph, int index) {
    boost::graph_traits<Graph>::vertex_iterator i, end;
    IndexMap index_map = get(boost::vertex_index, graph);
    
    for (tie(i, end) = vertices(graph); i != end; ++i) {
       // std::cout << name[get(index_map, *i)];
       std::cout << get(index_map, *i) << std::endl;
    }
    
    return false;
}

/**
 * Create and process the graph
 */
void boostSampleGraph() {

     /* actual graph structure  */
     Graph graph;

     /* add vertices to the graph  */
     add_vertex(Jeanie, graph);
     add_vertex(Debbie, graph);
     add_vertex(Rick, graph);
     add_vertex(John, graph);
     add_vertex(Amanda, graph);
     add_vertex(Margaret, graph);
     add_vertex(Benjamin, graph);
     // add_vertex(N, graph);
     
     /* add edges to the vertices in the graph*/
     add_edge(Jeanie, Debbie, edge_property(0.5f), graph);
     add_edge(Jeanie, Rick, edge_property(0.2f), graph);
     add_edge(Jeanie, John, edge_property(0.1f), graph);
     add_edge(Debbie, Amanda, edge_property(0.3f), graph);
     add_edge(Rick, Margaret, edge_property(0.4f), graph);
     add_edge(John, Benjamin, edge_property(0.6f), graph);
     // add_edge(Benjamin, Benjamin, edge_property(0.7f), graph);
     
     
     IndexMap index_map = get(boost::vertex_index, graph);
     EdgeDescriptor e_descriptor;
     EdgePropertyAccess edge_weights = get(boost::edge_weight, graph);
     VectorIterator i, end;
     
     bool continue_merging_flag = true;
     while (continue_merging_flag) {
        tie(i, end) = vertices(graph);
        AdjacencyIterator ai, a_end;
        tie(ai, a_end) = adjacent_vertices(*i, graph);

        for (; ai != a_end; ++ai) {
           bool found = false;
           tie(e_descriptor, found) = edge(*i, *ai, graph);
           if (found) {
              EdgeValue edge_val = boost::get(
                 boost::edge_weight, graph, e_descriptor);
              float weights_ = edge_val;
              if (weights_ > 0.3f) {
                 AdjacencyIterator aI, aEnd;
                 tie(aI, aEnd) = adjacent_vertices(*ai, graph);
                 for (; aI != aEnd; aI++) {
                    EdgeDescriptor ed;
                    bool located;
                    tie(ed, located) = edge(*aI, *ai, graph);
                    if (located && *aI != *i) {
                       add_edge(*i, *aI, graph);
                    }
                    std::cout << "\n DEBUG:  " << *i  << "  "
                              << *ai  << "  "
                              << *aI << " ";
                 }
                 std::cout << std::endl;
                 clear_vertex(*ai, graph);
                 remove_vertex(*ai, graph);
                  // std::cout << "Graph Size: "
                  //           << num_vertices(graph) << std::endl;
                 continue_merging_flag = updateRAG(graph, 0);
              }
           }
        }
     }
     
     
     for (tie(i, end) = vertices(graph); i != end; ++i) {
        std::cout << name[get(index_map, *i)];
        AdjacencyIterator ai, a_end;
        tie(ai, a_end) = adjacent_vertices(*i, graph);
        
        if (ai == a_end) {
           std::cout << " has no children";
        } else {
           std::cout << " is the parent of ";
        }
        for (; ai != a_end; ++ai) {
           bool found = false;
           tie(e_descriptor, found) = edge(*i, *ai, graph);
           if (found) {
              EdgeValue edge_val = boost::get(
                 boost::edge_weight, graph, e_descriptor);
              float weights_ = edge_val;
              if (weights_ > 0.3f) {
                 // - remove and merge
                 AdjacencyIterator aI, aEnd;
                 tie(aI, aEnd) = adjacent_vertices(*ai, graph);
                 for (; aI != aEnd; aI++) {
                    EdgeDescriptor ed;
                    bool located;
                    tie(ed, located) = edge(*aI, *ai, graph);
                    if (located && *aI != *i) {
                       add_edge(*i, *aI, graph);
                       // get(index_map, *i), get(index_map, *aI), graph);
                    }
                    
                    std::cout << "\n DEBUG:  " << *i  << "  "
                              << *ai  << "  "
                              << *aI << " ";
                    
                    
                    /* clear_vertex(*ai, graph);
                    // remove_vertex(*ai, graph);
                    
                    std::cout << "\n DEBUG2: " << *i  << "  "
                              << *ai  << "  "
                              << *aI << " ";
                    */
                 }
                 std::cout << std::endl;
                 clear_vertex(*ai, graph);
                 remove_vertex(*ai, graph);
                  // std::cout << "Graph Size: "
                  //           << num_vertices(graph) << std::endl;
                 updateRAG(graph, 0);
              }
           }
           // ai = tmp;
           // std::cout << name[get(index_map, *ai)];
           // if (boost::next(ai) != a_end) {
           //    std::cout << ", ";
           // }
        }
        std::cout << std::endl << std::endl;
     }
     std::cout << "\nGraph Size: " << num_vertices(graph) << std::endl;
}


/**
 * 
 */
int main(int argc, const char *argv[]) {

    boostSampleGraph();
    exit(-1);
   
   
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
