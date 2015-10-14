// Copyright (C) 2015 by 

#include <opencv2/opencv.hpp>
#include <iostream>

class UnionFind {
    struct Node {
       int parent;
       int rank;
    };
 private:
    Node *node;
   
 public:
    explicit UnionFind(int N);
    int findUF(int x);
    void unionUF(int x, int y);
};
/**
 * 
 */
UnionFind::UnionFind(int N) {
    this->node = new Node[N];
    for (int i = 0; i < N; i++) {
       this->node[i].parent = i;
       this->node[i].rank = 0;
    }
}
/**
 * 
 */
int UnionFind::findUF(int x) {

    if (x != this->node[x].parent) {
       this->node[x].parent = this->findUF(this->node[x].parent);
    }
    return this->node[x].parent;
}
/**
 * 
 */
void UnionFind::unionUF(int x, int y) {
    int xroot = this->findUF(x);
    int yroot = this->findUF(y);

    if (xroot == yroot) {
       return;
    }
    if (this->node[xroot].rank < this->node[yroot].rank) {
       this->node[xroot].parent = yroot;
    } else if (this->node[xroot].rank > this->node[yroot].rank) {
       this->node[yroot].parent = xroot;
    } else {
       this->node[yroot].parent = xroot;
       this->node[xroot].rank = ++this->node[xroot].rank;
    }
}

int main(int argc, char *argv[]) {

    
    return 0;
}

