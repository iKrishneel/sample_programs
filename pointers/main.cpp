

#include <iostream>

int main(int argc, char *argv[]) {
    int *p = new int;
    int q = *p;

    
    // q = 5;
    *p = 20;
    
    
    std::cout << *p << "\t" << q << std::endl;
    return 0;
}


