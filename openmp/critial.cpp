
#include <omp.h>
#include <iostream>

int main(int argc, char *argv[]) {
   
    int x;
    x = 0;

#pragma omp parallel shared(x)
    {
#pragma omp critical
       x = x + 1;
       std::cout << "THREAD: " << omp_get_thread_num() << " " << x
                 << std::endl;
    }  /* end of parallel section */

    std::cout << "\n" << x << std::endl;
}

/**
 * CRITICAL allows executation of one thread at a time
 */
