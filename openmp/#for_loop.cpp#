
#include <omp.h>
#include <iostream>

#define CHUNKSIZE 10
#define N     20

int main(int argc, char *argv[]) {

    int i, chunk;
    float a[N], b[N], c[N];

    /* Some initializations */
    for (i=0; i < N; i++)
       a[i] = b[i] = i * 1.0;
    chunk = CHUNKSIZE;

#pragma omp parallel shared(a, b, c, chunk) private(i)
    {
#pragma omp for schedule(dynamic, chunk) nowait
       
       for (i=0; i < N; i++) {
          std::cout << "\nIN FOR...." << i << std::endl;
          std::cout << " " << omp_get_thread_num() << std::endl;
          
          c[i] = a[i] + b[i];
       }
       
       // if (omp_get_thread_num() == 0) {
       // std::cout /*<< "NUM THREADS: " << omp_get_num_threads()*/
       //           << " " << omp_get_thread_num() << std::endl;
       // }
    }


    for (int j = 0; j < N; j++) {
       // std::cout << a[j] << " " << b[j]  << " " << c[j]
       //           << std::endl;
    }

    
}
