#pragma once

// Structural stand-in for the cuda runtime. Used ONLY by
// gpu/tools/precheck.py on machines without nvcc.
//
// It is permissive on purpose. Its job is to catch the mistakes that
// would otherwise burn the first paid minutes on a pod - typos,
// undeclared names, wrong argument counts at a kernel launch, unbalanced
// braces. It never judges cuda semantics, and it will happily accept an
// intrinsic that does not exist; the pod's nvcc is the real compiler.
// If it rejects code that nvcc accepts, widen the stub rather than
// bending the kernel to fit it.

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define __global__
#define __device__
#define __host__
#define __restrict__
#define __shared__
#define __forceinline__ inline
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
#define __noinline__

namespace vsg_stub {

// a value that converts to whatever the call site wants
struct any {
  operator int() const { return 0; }
  template <class T>
  operator T() const { return T(); }
};

// usable both as a value (__activemask) and as a call (__syncwarp())
struct anyfn {
  template <class... A>
  any operator()(A&&...) const { return any(); }
  operator int() const { return 0; }
};

}  // namespace vsg_stub

typedef enum { cudaSuccess = 0, cudaErrorUnknown = 999 } cudaError_t;
typedef enum {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
} cudaMemcpyKind;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUstream_st* cudaStream_t;

struct dim3 {
  unsigned int x, y, z;
  dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
      : x(x_), y(y_), z(z_) {}
};
extern dim3 blockIdx, threadIdx, blockDim, gridDim;

cudaError_t cudaMalloc(void** ptr, size_t bytes);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaMemset(void* dst, int value, size_t bytes);
cudaError_t cudaGetLastError();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
const char* cudaGetErrorString(cudaError_t error);

float __int_as_float(int bits);

// barriers, warp primitives, atomics: any arguments, any result
extern const vsg_stub::anyfn __syncthreads;
// shuffles return their argument's type, as the real ones do
template <class T>
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = 32);
template <class T>
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = 32);
template <class T>
T __shfl_xor_sync(unsigned mask, T var, int lane_mask, int width = 32);
template <class T>
T __shfl_sync(unsigned mask, T var, int src_lane, int width = 32);
unsigned __ballot_sync(unsigned mask, int predicate);
unsigned __activemask();
void __syncwarp(unsigned mask = 0xffffffffu);
template <class T>
T __reduce_min_sync(unsigned mask, T value);
template <class T>
T __reduce_max_sync(unsigned mask, T value);
template <class T>
T __reduce_add_sync(unsigned mask, T value);
template <class T>
T atomicCAS(T* address, T compare, T val);
template <class T>
T atomicAdd(T* address, T val);
template <class T>
T atomicMin(T* address, T val);
template <class T>
T atomicExch(T* address, T val);

// vector types and bit intrinsics used by the kernels
struct float2 {
  float x, y;
};
float2 make_float2(float x, float y);
int __ffs(int v);
int __ffs(unsigned int v);
int __popc(int v);
int __popc(unsigned int v);
