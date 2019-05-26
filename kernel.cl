__kernel void vecAdd(
    __global unsigned int* lcs,
    __constant char* x,
    unsigned int lcsWidth,
    __constant char* y,
    unsigned int lcsHeight
) {
    unsigned int numberOfIterations = lcsWidth + lcsHeight;
    const unsigned int i = get_global_id(0);
    for (unsigned int n = 0; n < numberOfIterations; n++) {
        const int j = n - i;
        if (i < lcsWidth && j >= 0 && j < lcsHeight) {
            const unsigned int offset = j * lcsWidth + i;

            unsigned int value;
            if (i == 0 || j == 0) {
                value = 0;
            } else {
                if (x[i - 1] == y[j - 1]) {
                    value = lcs[(j - 1) * lcsWidth + (i - 1)] + 1;
                } else {
                    unsigned int value1 = lcs[j * lcsWidth + (i - 1)];
                    unsigned int value2 = lcs[(j - 1) * lcsWidth + i];
                    value = value1 > value2 ? value1 : value2;
                }
            }

            lcs[offset] = value;
        }
        barrier(0);
    }
}