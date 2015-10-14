
#include <iostream>
#include <cmath>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <set>

int main(int argc, char *argv[]) {

    std::clock_t start;
    double duration;
    start = std::clock();
    
    const int aSize = 256000;
    double sinTable[aSize];

#pragma omp parallel
    {
#pragma omp for
    for (int n = 0; n < aSize; ++n) {
       sinTable[n] = std::sin(2 * M_PI * n / aSize);
    }
}

    duration = (std::clock() - start) /
       static_cast<double>(CLOCKS_PER_SEC);
    
    std::cout<<"printf: "<< duration <<'\n';
    return 0;
}

