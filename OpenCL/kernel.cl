__kernel void matrix_mul(__global const float *a, __global const float *b, __global float *c, unsigned int N, unsigned int K, unsigned int M) {
    // Naive multiplication of single cell

    // Get the index of the current element to be processed
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Do things

    float result = 0;
    for (int i = 0; i < K; i++) {
        result += a[x * K + i] * b[i * M + y];
    }
    c[x * M + y] = result;
}
