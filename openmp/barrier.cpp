
#include <omp.h>
#include <iostream>


void work(int n) {
    std::cout << n << std::endl;
}

void sub3(int n) {
    work(n);
    std::cout << "no barrier" << std::endl;
#pragma omp barrier
    {
       work(n);
       std::cout << "THREAD COUNT: " <<
          omp_get_thread_num() << "\t" << n << std::endl;
    }
}

void sub2(int k) {
#pragma omp parallel shared(k)
    {
       sub3(k);

       if (omp_get_thread_num() == 0) {
          std::cout << "sub2(.) Threads: " << omp_get_num_threads()
                    << std::endl;
       }
    }
}

void sub1(int n) {
    int i;
#pragma omp parallel private(i) shared(n)
    {
#pragma omp for  schedule(guided) nowait
       for (i = 0; i < n; i++) {
          sub2(i);
          // std::cout << omp_get_thread_num() << std::endl;
       }
       // if (omp_get_thread_num() == 0) {
       //    std::cout << "sub1(.) Threads: " << omp_get_num_threads()
       //              << std::endl;
       // }
    }
}

int main(int argc, char *argv[]) {
    // sub1(10);
    sub2(2);
    // sub3(2);

    return 0;
}


/**
 * The BARRIER will bind to the nearest PARALLEL region
 */
