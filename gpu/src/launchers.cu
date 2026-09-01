// Host-side glue for the gpu chapter: allocation, transfers, launch
// configuration, timing, and the cublas GEMM path for K1. The three
// hand-written kernels live in ../kernels and are pulled in textually
// below so the whole device side is one translation unit - no separate
// device linking to get wrong.
//
// The kernel launches for K1/K2/K3 are written out and compiled today,
// but each is preceded by a throw while its kernel is still a stub.
// That is on purpose: the compiler checks Sparsh's kernel signatures
// against these calls from day one, and enabling a kernel is a
// one-line deletion.

#include "launchers.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "config.cuh"

// one translation unit: the kernels are included, not linked
#include "hello.cu"
#include "distances.cu"
#include "topk.cu"
#include "hnsw_search.cu"

namespace vsg {
namespace {

#define VSG_CUDA(call)                                                    \
  do {                                                                    \
    cudaError_t vsg_err_ = (call);                                         \
    if (vsg_err_ != cudaSuccess) {                                         \
      throw std::runtime_error(std::string("cuda error at " __FILE__ ":") + \
                               std::to_string(__LINE__) + ": " +           \
                               cudaGetErrorString(vsg_err_));              \
    }                                                                      \
  } while (0)

#define VSG_CUBLAS(call)                                                   \
  do {                                                                     \
    cublasStatus_t vsg_st_ = (call);                                       \
    if (vsg_st_ != CUBLAS_STATUS_SUCCESS) {                                \
      throw std::runtime_error(std::string("cublas error at " __FILE__ ":") + \
                               std::to_string(__LINE__) + ": status " +     \
                               std::to_string((int)vsg_st_));               \
    }                                                                      \
  } while (0)

// a launch failure is asynchronous; this makes it a synchronous throw so
// the traceback points at the kernel that actually broke
void check_launch(const char* what) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("launch failed in ") + what + ": " +
                             cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("kernel failed in ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

// device buffer that frees itself, including when a later allocation in
// the same constructor throws
struct DevBuf {
  void* ptr;
  DevBuf() : ptr(nullptr) {}
  explicit DevBuf(size_t bytes) : ptr(nullptr) { alloc(bytes); }
  ~DevBuf() { free_(); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;

  void alloc(size_t bytes) {
    free_();
    if (bytes) VSG_CUDA(cudaMalloc(&ptr, bytes));
  }
  void free_() {
    if (ptr) {
      cudaFree(ptr);  // no throw: this runs from destructors
      ptr = nullptr;
    }
  }
  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr);
  }
};

// glue kernel: squared l2 norm of every row, fp32 accumulation. One
// block per row, blockDim must be a power of two.
__global__ void vsg_row_sqnorms(const __half* __restrict__ rows,
                                float* __restrict__ out, int stride, int dim) {
  extern __shared__ float sdata[];
  const __half* p = rows + (long long)blockIdx.x * stride;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = __half2float(p[i]);
    acc += v * v;
  }
  sdata[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sdata[0];
}

// glue kernel: finish the l2 identity after the GEMM left -2*q.b in out
__global__ void vsg_l2_epilogue(float* __restrict__ out,
                                const float* __restrict__ qnorms,
                                const float* __restrict__ bnorms, long long nq,
                                long long n) {
  long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long total = nq * n;
  if (idx >= total) return;
  long long qi = idx / n;
  long long bi = idx - qi * n;
  out[idx] += qnorms[qi] + bnorms[bi];
}

const int kNormThreads = 256;

// cap on the materialized fp32 distance matrix. batch 2048 x 1M would
// be 8 GB; chunking keeps a 24 GB card comfortable.
const long long kMaxMatrixBytes = 2LL * 1024 * 1024 * 1024;

int metric_code(const std::string& metric) {
  if (metric == "l2") return VSG_METRIC_L2;
  if (metric == "ip") return VSG_METRIC_IP;
  throw std::runtime_error(
      "unknown gpu metric '" + metric +
      "' - the device does l2 and ip only; normalize angular data at "
      "export and use ip (see gpu/SPEC.md)");
}

}  // namespace

struct DeviceIndex::Impl {
  DevBuf vectors;    // (n, stride) fp16
  DevBuf bnorms;     // (n,) fp32, l2 only
  DevBuf adjacency;  // (n, width) int32
  DevBuf degrees;    // (n,) int32
  long long n;
  long long dim;
  long long stride;
  long long width;
  int entry;
  int metric;
  bool has_graph;
  cublasHandle_t blas;
  cudaEvent_t ev_start;
  cudaEvent_t ev_stop;
  double last_ms;

  Impl()
      : n(0),
        dim(0),
        stride(0),
        width(0),
        entry(-1),
        metric(VSG_METRIC_L2),
        has_graph(false),
        blas(nullptr),
        ev_start(nullptr),
        ev_stop(nullptr),
        last_ms(0.0) {}

  void tic() { VSG_CUDA(cudaEventRecord(ev_start, 0)); }

  // milliseconds since tic(), added to the running total for this call
  void toc() {
    VSG_CUDA(cudaEventRecord(ev_stop, 0));
    VSG_CUDA(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    VSG_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    last_ms += (double)ms;
  }

  // fp32 squared norms of nq query rows, into a caller-owned buffer
  void query_sqnorms(const __half* queries, long long nq, float* out) {
    vsg_row_sqnorms<<<(int)nq, kNormThreads, kNormThreads * sizeof(float)>>>(
        queries, out, (int)stride, (int)dim);
    check_launch("vsg_row_sqnorms");
  }

  // K1 hand path, device pointers in and out
  void run_k1(const __half* queries, long long nq, float* out) {
    // DELETE-TO-ENABLE (K1): drop this throw once distances.cu is real
    throw std::runtime_error(
        "K1 not implemented - write gpu/kernels/distances.cu");

    dim3 block(VSG_K1_TILE, VSG_K1_TILE);
    dim3 grid((unsigned)((n + VSG_K1_TILE - 1) / VSG_K1_TILE),
              (unsigned)((nq + VSG_K1_TILE - 1) / VSG_K1_TILE));
    tic();
    vsg_k1_distances<<<grid, block>>>(queries, vectors.as<__half>(), out,
                                      (int)nq, (int)n, (int)dim, (int)stride,
                                      metric);
    check_launch("vsg_k1_distances");
    toc();
  }

  // cublas path, device pointers in and out
  void run_cublas(const __half* queries, long long nq, float* out) {
    // cublas is column-major. A row-major (r, c) matrix with row stride
    // ld reads, unchanged, as its own transpose: a column-major (c, r)
    // matrix with leading dimension ld. So:
    //   Q row-major (nq, stride) reads as Qc = Q^T, (dim x nq), ld=stride
    //   B row-major (n,  stride) reads as Bc = B^T, (dim x n),  ld=stride
    //   out row-major (nq, n)    reads as        (n x nq),      ld=n
    // we want out = Q . B^T, i.e. out^T = B . Q^T, and out^T is exactly
    // what cublas will write. B = (Bc)^T so op(A)=T on Bc; Q^T = Qc as
    // stored so op(B)=N on Qc. That gives
    //   C(n x nq) = op(Bc, T)(n x dim) . op(Qc, N)(dim x nq)
    // hence m=n, n=nq, k=dim, lda=ldb=stride, ldc=n.
    const float alpha = (metric == VSG_METRIC_L2) ? -2.0f : -1.0f;
    const float beta = 0.0f;
    tic();
    VSG_CUBLAS(cublasGemmEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)nq, (int)dim, &alpha,
        vectors.ptr, CUDA_R_16F, (int)stride, (const void*)queries, CUDA_R_16F,
        (int)stride, &beta, out, CUDA_R_32F, (int)n, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
    if (metric == VSG_METRIC_L2) {
      // out currently holds -2*q.b; add the two norm terms
      DevBuf qnorms(sizeof(float) * (size_t)nq);
      query_sqnorms(queries, nq, qnorms.as<float>());
      long long total = nq * n;
      int threads = 256;
      long long blocks = (total + threads - 1) / threads;
      vsg_l2_epilogue<<<(unsigned)blocks, threads>>>(
          out, qnorms.as<float>(), bnorms.as<float>(), nq, n);
      check_launch("vsg_l2_epilogue");
    }
    toc();
  }

  // K2, device pointers in and out
  void run_k2(const float* dmat, long long nq, int k, int* out_ids,
              float* out_dists) {
    // DELETE-TO-ENABLE (K2): drop this throw once topk.cu is real
    throw std::runtime_error("K2 not implemented - write gpu/kernels/topk.cu");

    tic();
    vsg_k2_topk<<<(unsigned)nq, VSG_K2_THREADS>>>(dmat, out_ids, out_dists,
                                                  (int)n, k);
    check_launch("vsg_k2_topk");
    toc();
  }
};

DeviceIndex::DeviceIndex(const uint16_t* vectors, long long n, long long stride,
                         long long dim, const std::string& metric)
    : p_(new Impl()) {
  try {
    if (n <= 0) throw std::runtime_error("index needs at least one vector");
    if (dim <= 0 || stride < dim) {
      throw std::runtime_error("bad dim/stride");
    }
    p_->n = n;
    p_->dim = dim;
    p_->stride = stride;
    p_->metric = metric_code(metric);

    size_t vbytes = sizeof(__half) * (size_t)n * (size_t)stride;
    p_->vectors.alloc(vbytes);
    VSG_CUDA(cudaMemcpy(p_->vectors.ptr, vectors, vbytes, cudaMemcpyHostToDevice));

    VSG_CUDA(cudaEventCreate(&p_->ev_start));
    VSG_CUDA(cudaEventCreate(&p_->ev_stop));
    VSG_CUBLAS(cublasCreate(&p_->blas));

    if (p_->metric == VSG_METRIC_L2) {
      // base norms never change, so pay for them once here instead of
      // on every query
      p_->bnorms.alloc(sizeof(float) * (size_t)n);
      p_->query_sqnorms(p_->vectors.as<__half>(), n, p_->bnorms.as<float>());
    }
  } catch (...) {
    // the DevBuf members free themselves; the handles may not exist yet
    if (p_->blas) cublasDestroy(p_->blas);
    if (p_->ev_start) cudaEventDestroy(p_->ev_start);
    if (p_->ev_stop) cudaEventDestroy(p_->ev_stop);
    delete p_;
    p_ = nullptr;
    throw;
  }
}

DeviceIndex::~DeviceIndex() {
  if (!p_) return;
  if (p_->blas) cublasDestroy(p_->blas);
  if (p_->ev_start) cudaEventDestroy(p_->ev_start);
  if (p_->ev_stop) cudaEventDestroy(p_->ev_stop);
  delete p_;
  p_ = nullptr;
}

long long DeviceIndex::n() const { return p_->n; }
long long DeviceIndex::dim() const { return p_->dim; }
long long DeviceIndex::stride() const { return p_->stride; }
bool DeviceIndex::has_graph() const { return p_->has_graph; }
double DeviceIndex::last_kernel_ms() const { return p_->last_ms; }

void DeviceIndex::set_graph(const int* adjacency, const int* degrees,
                            long long n, long long width, int entry) {
  if (n != p_->n) {
    throw std::runtime_error("graph node count does not match the vectors");
  }
  if (width <= 0) throw std::runtime_error("adjacency width must be positive");
  if (entry < 0 || entry >= n) throw std::runtime_error("entry out of range");

  size_t abytes = sizeof(int) * (size_t)n * (size_t)width;
  p_->adjacency.alloc(abytes);
  VSG_CUDA(cudaMemcpy(p_->adjacency.ptr, adjacency, abytes,
                      cudaMemcpyHostToDevice));
  p_->degrees.alloc(sizeof(int) * (size_t)n);
  VSG_CUDA(cudaMemcpy(p_->degrees.ptr, degrees, sizeof(int) * (size_t)n,
                      cudaMemcpyHostToDevice));
  p_->width = width;
  p_->entry = entry;
  p_->has_graph = true;
}

void DeviceIndex::distances(const uint16_t* queries, long long nq, float* out) {
  p_->last_ms = 0.0;
  size_t qbytes = sizeof(__half) * (size_t)nq * (size_t)p_->stride;
  DevBuf dq(qbytes);
  VSG_CUDA(cudaMemcpy(dq.ptr, queries, qbytes, cudaMemcpyHostToDevice));
  DevBuf dout(sizeof(float) * (size_t)nq * (size_t)p_->n);
  p_->run_k1(dq.as<__half>(), nq, dout.as<float>());
  VSG_CUDA(cudaMemcpy(out, dout.ptr, sizeof(float) * (size_t)nq * (size_t)p_->n,
                      cudaMemcpyDeviceToHost));
}

void DeviceIndex::distances_cublas(const uint16_t* queries, long long nq,
                                   float* out) {
  p_->last_ms = 0.0;
  size_t qbytes = sizeof(__half) * (size_t)nq * (size_t)p_->stride;
  DevBuf dq(qbytes);
  VSG_CUDA(cudaMemcpy(dq.ptr, queries, qbytes, cudaMemcpyHostToDevice));
  DevBuf dout(sizeof(float) * (size_t)nq * (size_t)p_->n);
  p_->run_cublas(dq.as<__half>(), nq, dout.as<float>());
  VSG_CUDA(cudaMemcpy(out, dout.ptr, sizeof(float) * (size_t)nq * (size_t)p_->n,
                      cudaMemcpyDeviceToHost));
}

void DeviceIndex::brute_force(const uint16_t* queries, long long nq, int k,
                              bool use_cublas, int* out_ids, float* out_dists) {
  if (k < 1 || k > VSG_MAX_K) {
    throw std::runtime_error("k must be in [1, " + std::to_string(VSG_MAX_K) +
                             "] - the top-k kernel's per-thread lists cap it");
  }
  if ((long long)k > p_->n) {
    throw std::runtime_error("k is larger than the index");
  }
  p_->last_ms = 0.0;

  // one chunk of queries at a time, so the fp32 matrix stays under the cap
  long long per_row = p_->n * (long long)sizeof(float);
  long long chunk = kMaxMatrixBytes / per_row;
  if (chunk < 1) chunk = 1;
  if (chunk > nq) chunk = nq;

  DevBuf dq(sizeof(__half) * (size_t)chunk * (size_t)p_->stride);
  DevBuf dmat(sizeof(float) * (size_t)chunk * (size_t)p_->n);
  DevBuf dids(sizeof(int) * (size_t)nq * (size_t)k);
  DevBuf ddists(sizeof(float) * (size_t)nq * (size_t)k);

  for (long long q0 = 0; q0 < nq; q0 += chunk) {
    long long rows = nq - q0;
    if (rows > chunk) rows = chunk;
    size_t qbytes = sizeof(__half) * (size_t)rows * (size_t)p_->stride;
    VSG_CUDA(cudaMemcpy(dq.ptr, queries + q0 * p_->stride, qbytes,
                        cudaMemcpyHostToDevice));
    if (use_cublas) {
      p_->run_cublas(dq.as<__half>(), rows, dmat.as<float>());
    } else {
      p_->run_k1(dq.as<__half>(), rows, dmat.as<float>());
    }
    p_->run_k2(dmat.as<float>(), rows, k, dids.as<int>() + q0 * k,
               ddists.as<float>() + q0 * k);
  }
  VSG_CUDA(cudaMemcpy(out_ids, dids.ptr, sizeof(int) * (size_t)nq * (size_t)k,
                      cudaMemcpyDeviceToHost));
  VSG_CUDA(cudaMemcpy(out_dists, ddists.ptr,
                      sizeof(float) * (size_t)nq * (size_t)k,
                      cudaMemcpyDeviceToHost));
}

void DeviceIndex::hnsw(const uint16_t* queries, long long nq, const int* entries,
                       int k, int ef, int* out_ids, float* out_dists) {
  if (!p_->has_graph) {
    throw std::runtime_error("call set_graph() before hnsw()");
  }
  if (ef < 1 || ef > VSG_MAX_EF) {
    throw std::runtime_error("ef must be in [1, " + std::to_string(VSG_MAX_EF) +
                             "]");
  }
  if (k < 1 || k > ef) throw std::runtime_error("k must be in [1, ef]");
  if (k > VSG_MAX_K) {
    throw std::runtime_error("k must be <= " + std::to_string(VSG_MAX_K));
  }
  p_->last_ms = 0.0;

  size_t qbytes = sizeof(__half) * (size_t)nq * (size_t)p_->stride;
  DevBuf dq(qbytes);
  VSG_CUDA(cudaMemcpy(dq.ptr, queries, qbytes, cudaMemcpyHostToDevice));
  DevBuf dentries(sizeof(int) * (size_t)nq);
  VSG_CUDA(cudaMemcpy(dentries.ptr, entries, sizeof(int) * (size_t)nq,
                      cudaMemcpyHostToDevice));
  DevBuf dids(sizeof(int) * (size_t)nq * (size_t)k);
  DevBuf ddists(sizeof(float) * (size_t)nq * (size_t)k);

  // one visited row per in-flight query. -1 is the empty marker and
  // 0xFF bytes are exactly -1 in two's complement, so memset fills it.
  size_t vbytes = sizeof(int) * (size_t)nq * (size_t)VSG_VISITED_SLOTS;
  DevBuf visited(vbytes);
  VSG_CUDA(cudaMemset(visited.ptr, 0xFF, vbytes));

  // DELETE-TO-ENABLE (K3): drop this throw once hnsw_search.cu is real
  throw std::runtime_error(
      "K3 not implemented - write gpu/kernels/hnsw_search.cu");

  int threads = VSG_K3_WARPS_PER_BLOCK * 32;
  long long blocks = (nq + VSG_K3_WARPS_PER_BLOCK - 1) / VSG_K3_WARPS_PER_BLOCK;
  int smem = VSG_K3_SMEM_PER_WARP(ef) * VSG_K3_WARPS_PER_BLOCK;
  p_->tic();
  vsg_k3_hnsw_search<<<(unsigned)blocks, threads, smem>>>(
      p_->vectors.as<__half>(), p_->adjacency.as<int>(), p_->degrees.as<int>(),
      dq.as<__half>(), dentries.as<int>(), visited.as<int>(), dids.as<int>(),
      ddists.as<float>(), (int)nq, (int)p_->n, (int)p_->dim, (int)p_->stride,
      (int)p_->width, k, ef, p_->metric);
  check_launch("vsg_k3_hnsw_search");
  p_->toc();

  VSG_CUDA(cudaMemcpy(out_ids, dids.ptr, sizeof(int) * (size_t)nq * (size_t)k,
                      cudaMemcpyDeviceToHost));
  VSG_CUDA(cudaMemcpy(out_dists, ddists.ptr,
                      sizeof(float) * (size_t)nq * (size_t)k,
                      cudaMemcpyDeviceToHost));
}

int max_k() { return VSG_MAX_K; }
int max_ef() { return VSG_MAX_EF; }

void hello_add(const float* a, const float* b, float* out, long long n) {
  size_t bytes = sizeof(float) * (size_t)n;
  DevBuf da(bytes), db(bytes), dout(bytes);
  VSG_CUDA(cudaMemcpy(da.ptr, a, bytes, cudaMemcpyHostToDevice));
  VSG_CUDA(cudaMemcpy(db.ptr, b, bytes, cudaMemcpyHostToDevice));
  int threads = 256;
  long long blocks = (n + threads - 1) / threads;
  vsg_hello_add<<<(unsigned)blocks, threads>>>(da.as<float>(), db.as<float>(),
                                               dout.as<float>(), n);
  check_launch("vsg_hello_add");
  VSG_CUDA(cudaMemcpy(out, dout.ptr, bytes, cudaMemcpyDeviceToHost));
}

void topk(const float* dmat, long long nq, long long n, int k, int* out_ids,
          float* out_dists, double* kernel_ms) {
  if (k < 1 || k > VSG_MAX_K) {
    throw std::runtime_error("k must be in [1, " + std::to_string(VSG_MAX_K) +
                             "] - the top-k kernel's per-thread lists cap it");
  }
  if ((long long)k > n) throw std::runtime_error("k is larger than the row");

  DevBuf dmat_dev(sizeof(float) * (size_t)nq * (size_t)n);
  VSG_CUDA(cudaMemcpy(dmat_dev.ptr, dmat, sizeof(float) * (size_t)nq * (size_t)n,
                      cudaMemcpyHostToDevice));
  DevBuf dids(sizeof(int) * (size_t)nq * (size_t)k);
  DevBuf ddists(sizeof(float) * (size_t)nq * (size_t)k);

  // DELETE-TO-ENABLE (K2): drop this throw once topk.cu is real
  throw std::runtime_error("K2 not implemented - write gpu/kernels/topk.cu");

  cudaEvent_t start, stop;
  VSG_CUDA(cudaEventCreate(&start));
  VSG_CUDA(cudaEventCreate(&stop));
  VSG_CUDA(cudaEventRecord(start, 0));
  vsg_k2_topk<<<(unsigned)nq, VSG_K2_THREADS>>>(
      dmat_dev.as<float>(), dids.as<int>(), ddists.as<float>(), (int)n, k);
  check_launch("vsg_k2_topk");
  VSG_CUDA(cudaEventRecord(stop, 0));
  VSG_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  VSG_CUDA(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (kernel_ms) *kernel_ms = (double)ms;

  VSG_CUDA(cudaMemcpy(out_ids, dids.ptr, sizeof(int) * (size_t)nq * (size_t)k,
                      cudaMemcpyDeviceToHost));
  VSG_CUDA(cudaMemcpy(out_dists, ddists.ptr,
                      sizeof(float) * (size_t)nq * (size_t)k,
                      cudaMemcpyDeviceToHost));
}

}  // namespace vsg
