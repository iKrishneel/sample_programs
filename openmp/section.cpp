
#include <omp.h>
#include <iostream>

#define N 20

int main(int argc, char *argv[]) {
    int i;
    float a[N], b[N], c[N], d[N];

/* Some initializations */
    for (i=0; i < N; i++) {
       a[i] = i * 1.5;
       b[i] = i + 22.35;
    }

#pragma omp parallel shared(a, b, c, d) private(i)
    {
#pragma omp sections nowait
       {
#pragma omp section
          for (i=0; i < N; i++) {
             std::cout << "THREAD S1: " << omp_get_thread_num() << std::endl;
             c[i] = a[i] + b[i];
          }
          
#pragma omp section
          for (i=0; i < N; i++) {
             std::cout << "THREAD S2: " << omp_get_thread_num() << std::endl;
             d[i] = a[i] * b[i];
          }

       }  /* end of sections */

    }  /* end of parallel section */
}
