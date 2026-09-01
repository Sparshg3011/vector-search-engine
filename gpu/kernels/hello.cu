// Toolchain check. Not part of the port - this exists so phase-0 can
// prove the build, the driver, ncu and compute-sanitizer all work
// before any real kernel is written.

__global__ void vsg_hello_add(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
