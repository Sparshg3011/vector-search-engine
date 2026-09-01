#pragma once

// Host-side API of the gpu chapter. Deliberately free of cuda types so
// bindings.cpp compiles with the host compiler and never needs the cuda
// headers - everything device-side hides behind DeviceIndex::Impl.
//
// fp16 crosses this boundary as uint16_t: numpy hands us float16 data
// and pybind11 has no half type, so the pointer is reinterpreted (never
// converted) on the way in.

#include <cstdint>
#include <string>

namespace vsg {

// Kernel capacity limits, mirrored out of kernels/config.cuh so the
// bindings can reject bad arguments as ValueError before anything
// touches the device. config.cuh stays the single source of truth.
int max_k();
int max_ef();

// Toolchain check: out = a + b on the device.
void hello_add(const float* a, const float* b, float* out, long long n);

// k2 standalone, so top-k is testable without a stored index. dmat is
// row-major (nq, n) fp32 on the host; rows come back sorted ascending.
// kernel_ms receives the cuda-event time of the kernel alone.
void topk(const float* dmat, long long nq, long long n, int k, int* out_ids,
          float* out_dists, double* kernel_ms);

class DeviceIndex {
 public:
  // vectors is row-major (n, stride) fp16, rows padded with zeros out to
  // stride. metric is "l2" or "ip".
  DeviceIndex(const uint16_t* vectors, long long n, long long stride,
              long long dim, const std::string& metric);
  ~DeviceIndex();
  DeviceIndex(const DeviceIndex&) = delete;
  DeviceIndex& operator=(const DeviceIndex&) = delete;

  // layer-0 graph: adjacency (n, width) int32 padded with -1, degrees
  // (n,), and the entry point. Required before hnsw().
  void set_graph(const int* adjacency, const int* degrees, long long n,
                 long long width, int entry);

  // k1 alone, hand-rolled path: out is (nq, n) fp32. The caller keeps nq
  // small - this materializes the whole matrix with no chunking.
  void distances(const uint16_t* queries, long long nq, float* out);

  // k1 alone, cublas GEMM path. Same shapes.
  void distances_cublas(const uint16_t* queries, long long nq, float* out);

  // k1 + k2. Chunks internally so the materialized fp32 matrix stays
  // under the 2 GB cap (batch 2048 x 1M would be 8 GB).
  void brute_force(const uint16_t* queries, long long nq, int k,
                   bool use_cublas, int* out_ids, float* out_dists);

  // k3. entries holds one layer-0 start node per query, from the host
  // descent in vecstore_gpu.descend().
  void hnsw(const uint16_t* queries, long long nq, const int* entries, int k,
            int ef, int* out_ids, float* out_dists);

  // cuda-event time of the kernels in the previous call, milliseconds.
  // Excludes host-device copies on purpose: benchmarks report kernel and
  // wall time as separate columns.
  double last_kernel_ms() const;

  long long n() const;
  long long dim() const;
  long long stride() const;
  bool has_graph() const;

 private:
  struct Impl;
  Impl* p_;
};

}  // namespace vsg
