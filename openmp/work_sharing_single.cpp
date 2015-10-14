
#include <omp.h>
#include <iostream>

#define N 20
#define CHUNKSIZE 2

int main(int argc, char *argv[]) {

    int i, chunk;
    float a[N], b[N], c[N];
    
    for (i=0; i < N; i++) {
       a[i] = b[i] = i * 1.0;
    }
    chunk = CHUNKSIZE;

    int val = 0;
#pragma omp parallel \
   for shared(a, b, c, chunk, val) private(i)       \
   schedule(static, chunk) num_threads(8)
    for (i=0; i < N; i++) {
       std::cout << "THREAD: " << omp_get_thread_num() << "\t"
                 << omp_get_num_threads()
                 << std::endl;
       c[i] = a[i] + b[i];
       val++;
    }
    std::cout << "Val: " << val << std::endl;
}
