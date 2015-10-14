
#include <omp.h>
#include <iostream>

float work1(int i) {
    return 1.0 * i;
}

float work2(int i) {
    return 2.0 * i;
}

void atomic_example(float *x, float *y, int *index, int n) {
    int i;

#pragma omp parallel shared(x, y, index, n) private(i)
    {
#pragma omp for schedule(guided)
       for (i = 0; i < n; i++) {
#pragma omp atomic
          x[index[i]] += work1(i);
          y[i] += work2(i);

          std::cout << i << std::endl;
       }
       std::cout << omp_get_thread_num() << std::endl;
    }
    
}

int main(int argc, char *argv[]) {
    float x[1000];
    float y[1000];
    int index[1000];
    int i;

    int N = 10;
    for (i = 0; i < N; i++) {
       index[i] = i %N;
       y[i] = 0.0;
    }
    for (i = 0; i < N; i++) {
       x[i] = 0.0;
    }
    atomic_example(x, y, index, N);

    for (i = 0; i < N; i++) {
       std::cout << x[i] << "\t" << y[i] << "\t" << index[i]<< std::endl;
    }
    
    return 0;
}

