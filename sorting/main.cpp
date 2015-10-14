
#include <opencv2/core/core.hpp>

#include <iostream>
#include <string>
// #include <multimap>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>

void sortvector() {
    std::vector<std::pair<int, std::vector<int> > > list;
    
    std::vector<int> a;
    a.push_back(1);
    a.push_back(5);

    std::vector<int> b;
    b.push_back(1);
    b.push_back(5);
    b.push_back(6);

    std::vector<int> c;
    c.push_back(2);
    c.push_back(3);
    c.push_back(4);

    std::vector<int> d;
    d.push_back(1);

    std::vector<int> e;
    e.push_back(3);

    std::vector<int> f;
    f.push_back(2);
    f.push_back(3);
    
    list.push_back(std::pair<int, std::vector<int> >(2, a));
    list.push_back(std::pair<int, std::vector<int> >(3, b));
    list.push_back(std::pair<int, std::vector<int> >(1, c));
    list.push_back(std::pair<int, std::vector<int> >(4, d));
    list.push_back(std::pair<int, std::vector<int> >(6, e));
    list.push_back(std::pair<int, std::vector<int> >(5, f));
    
    std::sort(list.begin(), list.end());

    for (int i = 0; i < list.size(); i++) {
       std::cout << list[i].first << "\t< ";
       
       for (int j = 0; j < list[i].second.size(); j++) {
          std::cout << list[i].second.at(j) << ", ";
       }
       std::cout << " >\n" << std::endl;
    }
}

int main(int argc, char *argv[]) {

    std::map<int, std::vector<int> > list;
    std::vector<int> a;
    a.push_back(1);
    a.push_back(5);

    std::vector<int> b;
    b.push_back(1);
    b.push_back(5);
    b.push_back(6);

    std::vector<int> c;
    c.push_back(2);
    c.push_back(3);
    c.push_back(4);

    std::vector<int> d;
    d.push_back(1);

    std::vector<int> e;
    e.push_back(3);

    std::vector<int> f;
    f.push_back(2);
    f.push_back(3);

    list[2] = a;
    // list[3] = b;
    // list[1] = c;
    // list[4] = d;
    // list[6] = e;
    // list[5] = f;


    int j = 2;
    for (int i = 0; i < list.find(j)->second.size(); i++) {
       int val = list.find(j)->second.at(i);
       std::cout << val << " ";
    }
    std::cout << "\n" << std::endl;
    
    return 0;
}

