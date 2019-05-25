__kernel void vecAdd(__global float* a)
{
    int gid = get_global_id(0);// in CUDA = blockIdx.x * blockDim.x + threadIdx.x

    a[gid] += a[gid];
}