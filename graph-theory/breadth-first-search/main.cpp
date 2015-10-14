
#include <iostream>

/**
 * Tree structure
 */
struct Node {
    int info;  // data on this node
    Node *next;  // Pointer to the next subtree
};

class Graph {
 private:
    int n;  // number of vertices in the graph
    int **A;
   
 public:
    explicit Graph(const int = 2);
    bool isConnected(int, int);
    void addEdge(int, int);
    void BFS(int);
};

/**
 * constructor
 */
Graph::Graph(const int size) {
    int i, j;
    if (size < 2) {
       this->n = 2;
    }
    this->n = size;
    this->A = new int*[this->n];

    for (int i = 0; i < this->n; i++) {
       this->A[i] = new int[this->n];
    }
    for (int j = 0; j < this->n; j++) {
       for (int i = 0; i < this->n; i++) {
          this->A[j][i] = 0;
       }
    }
}

/**
 * check if two verices are connected by an edge
 @param u vertex
 @oaran v vertex
 */
bool Graph::isConnected(int u, int v) {
    return(this->A[u-1][v-1] == sizeof(char));
}

/**
 * add an edge E to the graph G
 */
void Graph::addEdge(int u, int v) {
    this->A[u-1][v-1] = this->A[v-1][u-1] = sizeof(char);
}

/**
 * main function
 */
int main(int argc, char *argv[]) {

    
    return 0;
}
