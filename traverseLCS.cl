__kernel void traverseLCS(
    __global unsigned int* lcs,
    __constant char* x,
    unsigned int lcsWidth,
    __constant char* y,
    unsigned int lcsHeight,
    __global void* results
) {
    unsigned int numberOfIterations = lcsWidth + lcsHeight;
    unsigned int lcsSize = lcsWidth * lcsHeight;
    unsigned int resultStringLength = lcs[lcsSize - 1];
    unsigned int resultCStringSize = resultStringLength + 1;
    const size_t resultCellSize = sizeof(unsigned int) + resultStringLength * resultStringLength * resultCStringSize;
    const unsigned int i = get_global_id(0);

    for (unsigned int n = 0; n < numberOfIterations; n++) {
        const int j = n - i;
        if (i < lcsWidth && j >= 0 && j < lcsHeight) {
            const unsigned int lcsOffset = j * lcsWidth + i;
            const unsigned int resultsOffset = lcsOffset * resultCellSize;
            results[resultsOffset] = n;
        }
        barrier(0);
    }
}