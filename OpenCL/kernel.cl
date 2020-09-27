__kernel void matrix_mul(__global const float *a, __global const float *b, __global float *c, unsigned int N, unsigned int K, unsigned int M) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int gr = get_global_id(0);
    const int gc = get_global_id(1);

    __local float a_local[16][16];
    __local float b_local[16][16];

    float current = 0;

    const int tiles = K / 16;
    for (int t = 0; t < tiles; t++) {
        const int tiled_row = 16 * t + row;
        const int tiled_col = 16 * t + col;
        a_local[col][row] = a[gr * K + tiled_col];
        b_local[col][row] = b[tiled_row * M + gc];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; k++) {
            current += a_local[k][row] * b_local[col][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[gr * M + gc] = current;
}
