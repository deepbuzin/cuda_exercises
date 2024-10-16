#include <stdio.h>
#include <stdlib.h>

#define N 4


void cpu_add(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}


void init_vector(float *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = i;
    }
}


int main() {
    float *h_a, *h_b, *h_c;
    size_t size = N * sizeof(float);

    h_a = (float*) malloc(size);
    h_b = (float*) malloc(size);
    h_c = (float*) malloc(size);

    init_vector(h_a, N);
    init_vector(h_b, N);

    cpu_add(h_a, h_b, h_c, N);

    for (int i = 0; i < N; i++) {
        printf("%.0f ", h_c[i]);
    }

}




