__kernel void calculateLCS(
    __global unsigned long* lcs,
    __constant char* x,
    unsigned long lcsWidth,
    __constant char* y,
    unsigned long lcsHeight
) {
    unsigned long numberOfIterations = lcsWidth + lcsHeight - 1;
    const unsigned long i = get_global_id(0);
    for (unsigned long n = 0; n < numberOfIterations; n++) {
        const int j = n - i;
        if (i < lcsWidth && j >= 0 && j < lcsHeight) {
            const unsigned long offset = j * lcsWidth + i;

            unsigned long value;
            if (i == 0 || j == 0) {
                value = 0;
            } else {
                if (x[i - 1] == y[j - 1]) {
                    value = lcs[(j - 1) * lcsWidth + (i - 1)] + 1;
                } else {
                    unsigned long value1 = lcs[j * lcsWidth + (i - 1)];
                    unsigned long value2 = lcs[(j - 1) * lcsWidth + i];
                    value = value1 > value2 ? value1 : value2;
                }
            }

            lcs[offset] = value;
        }
        barrier(0);
    }
}