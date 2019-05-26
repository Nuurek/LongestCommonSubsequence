__kernel void vecAdd(__global unsigned int* a)
{
    int gid = get_global_id(0);// in CUDA = blockIdx.x * blockDim.x + threadIdx.x

    a[gid] = gid;
}