// pybind11 surface of the gpu chapter. Compiled by the host compiler:
// launchers.h is deliberately cuda-free, so nothing here needs the cuda
// headers.
//
// fp16 is the awkward part. numpy hands us float16 and pybind11 has no
// half type, so these arrays arrive as a generic py::array whose dtype
// we check by hand (kind 'f', itemsize 2) and whose buffer we
// REINTERPRET as uint16_t. Taking py::array_t<uint16_t> instead would
// let numpy silently *convert* the values to integers, which looks like
// a working call and returns garbage.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "launchers.h"

namespace py = pybind11;

namespace {

// validates a C-contiguous 2-d float16 array and hands back its buffer
const uint16_t* half_2d(const py::array& a, const char* name, py::ssize_t& rows,
                        py::ssize_t& cols) {
  if (a.ndim() != 2) {
    throw py::value_error(std::string(name) + " must be 2-d, got ndim " +
                          std::to_string(a.ndim()));
  }
  if (a.dtype().kind() != 'f' || a.itemsize() != 2) {
    throw py::value_error(std::string(name) +
                          " must be float16 (see gpu/SPEC.md); export.py "
                          "writes the .gpu.npz in that dtype");
  }
  if (!(a.flags() & py::array::c_style)) {
    throw py::value_error(std::string(name) +
                          " must be C-contiguous - use np.ascontiguousarray");
  }
  rows = a.shape(0);
  cols = a.shape(1);
  return static_cast<const uint16_t*>(a.data());
}

// validates a C-contiguous 2-d float32 array. Deliberately strict: a
// forcecast here would silently downcast an fp64 matrix, which changes
// which ids win at the top-k boundary and looks like a working call.
const float* float_2d(const py::array& a, const char* name, py::ssize_t& rows,
                      py::ssize_t& cols) {
  if (a.ndim() != 2) {
    throw py::value_error(std::string(name) + " must be 2-d, got ndim " +
                          std::to_string(a.ndim()));
  }
  if (a.dtype().kind() != 'f' || a.itemsize() != 4) {
    throw py::value_error(std::string(name) +
                          " must be float32 - k2 consumes the fp32 matrix k1 "
                          "produced, and a silent downcast would move the "
                          "top-k boundary");
  }
  if (!(a.flags() & py::array::c_style)) {
    throw py::value_error(std::string(name) +
                          " must be C-contiguous - use np.ascontiguousarray");
  }
  rows = a.shape(0);
  cols = a.shape(1);
  return static_cast<const float*>(a.data());
}

void check_positive(py::ssize_t v, const char* what) {
  if (v <= 0) {
    throw py::value_error(std::string(what) + " must be positive");
  }
}

// argument checks live here, ahead of the device call, so bad input is a
// ValueError rather than the RuntimeError the c++ core raises as a
// backstop (gpu/SPEC.md)
void check_k(int k, py::ssize_t n) {
  if (k < 1) throw py::value_error("k must be >= 1");
  if (k > vsg::max_k()) {
    throw py::value_error("k must be <= " + std::to_string(vsg::max_k()) +
                          " - the top-k kernel's per-thread lists cap it");
  }
  if ((py::ssize_t)k > n) {
    throw py::value_error("k=" + std::to_string(k) +
                          " is larger than the " + std::to_string(n) +
                          " candidates available");
  }
}

class PyDeviceIndex {
 public:
  PyDeviceIndex(const py::array& vectors, py::ssize_t dim,
                const std::string& metric) {
    py::ssize_t n = 0, stride = 0;
    const uint16_t* p = half_2d(vectors, "vectors", n, stride);
    check_positive(n, "vector count");
    check_positive(dim, "dim");
    if (dim > stride) {
      throw py::value_error("dim " + std::to_string(dim) +
                            " exceeds the row stride " + std::to_string(stride));
    }
    if (metric != "l2" && metric != "ip") {
      throw py::value_error(
          "unknown gpu metric '" + metric +
          "' - the device does l2 and ip only; normalize angular data at "
          "export and use ip (see gpu/SPEC.md)");
    }
    stride_ = stride;
    n_ = n;
    py::gil_scoped_release release;
    index_ = new vsg::DeviceIndex(p, n, stride, dim, metric);
  }

  ~PyDeviceIndex() { delete index_; }
  PyDeviceIndex(const PyDeviceIndex&) = delete;
  PyDeviceIndex& operator=(const PyDeviceIndex&) = delete;

  void set_graph(const py::array_t<int32_t, py::array::c_style |
                                                py::array::forcecast>& adjacency,
                 const py::array_t<int32_t, py::array::c_style |
                                                py::array::forcecast>& degrees,
                 int entry) {
    if (adjacency.ndim() != 2) throw py::value_error("adjacency must be 2-d");
    if (degrees.ndim() != 1) throw py::value_error("degrees must be 1-d");
    if (adjacency.shape(0) != n_ || degrees.shape(0) != n_) {
      throw py::value_error("adjacency/degrees rows must match the vectors");
    }
    py::ssize_t width = adjacency.shape(1);
    const int32_t* a = adjacency.data();
    const int32_t* d = degrees.data();
    py::gil_scoped_release release;
    index_->set_graph(a, d, n_, width, entry);
  }

  py::array_t<float> distances(const py::array& queries, bool use_cublas) {
    py::ssize_t nq = 0, cols = 0;
    const uint16_t* q = half_2d(queries, "queries", nq, cols);
    check_positive(nq, "query count");
    if (cols != stride_) {
      throw py::value_error("queries must have the index row stride " +
                            std::to_string(stride_) + ", got " +
                            std::to_string(cols));
    }
    std::vector<py::ssize_t> shape;
    shape.push_back(nq);
    shape.push_back((py::ssize_t)n_);
    py::array_t<float> out(shape);
    float* o = out.mutable_data();
    {
      py::gil_scoped_release release;
      if (use_cublas) {
        index_->distances_cublas(q, nq, o);
      } else {
        index_->distances(q, nq, o);
      }
    }
    return out;
  }

  py::tuple brute_force(const py::array& queries, int k, bool use_cublas) {
    py::ssize_t nq = 0, cols = 0;
    const uint16_t* q = half_2d(queries, "queries", nq, cols);
    check_positive(nq, "query count");
    if (cols != stride_) {
      throw py::value_error("queries must have the index row stride " +
                            std::to_string(stride_) + ", got " +
                            std::to_string(cols));
    }
    check_k(k, n_);
    std::vector<py::ssize_t> shape;
    shape.push_back(nq);
    shape.push_back((py::ssize_t)k);
    py::array_t<int32_t> ids(shape);
    py::array_t<float> dists(shape);
    int32_t* oi = ids.mutable_data();
    float* od = dists.mutable_data();
    {
      py::gil_scoped_release release;
      index_->brute_force(q, nq, k, use_cublas, oi, od);
    }
    return py::make_tuple(ids, dists);
  }

  py::tuple hnsw(const py::array& queries,
                 const py::array_t<int32_t, py::array::c_style |
                                                py::array::forcecast>& entries,
                 int k, int ef) {
    py::ssize_t nq = 0, cols = 0;
    const uint16_t* q = half_2d(queries, "queries", nq, cols);
    check_positive(nq, "query count");
    if (cols != stride_) {
      throw py::value_error("queries must have the index row stride " +
                            std::to_string(stride_) + ", got " +
                            std::to_string(cols));
    }
    if (entries.ndim() != 1 || entries.shape(0) != nq) {
      throw py::value_error("entries must be 1-d with one start node per query");
    }
    if (ef < 1 || ef > vsg::max_ef()) {
      throw py::value_error(
          "ef must be in [1, " + std::to_string(vsg::max_ef()) +
          "] - the candidate list lives in shared memory and is sized from it");
    }
    check_k(k, n_);
    if (k > ef) {
      throw py::value_error("k must be <= ef - a beam of width " +
                            std::to_string(ef) +
                            " cannot hand back " + std::to_string(k) +
                            " results");
    }
    const int32_t* e = entries.data();
    std::vector<py::ssize_t> shape;
    shape.push_back(nq);
    shape.push_back((py::ssize_t)k);
    py::array_t<int32_t> ids(shape);
    py::array_t<float> dists(shape);
    int32_t* oi = ids.mutable_data();
    float* od = dists.mutable_data();
    {
      py::gil_scoped_release release;
      index_->hnsw(q, nq, e, k, ef, oi, od);
    }
    return py::make_tuple(ids, dists);
  }

  double last_kernel_ms() const { return index_->last_kernel_ms(); }
  py::ssize_t size() const { return n_; }
  py::ssize_t dim() const { return index_->dim(); }
  py::ssize_t stride() const { return stride_; }
  bool has_graph() const { return index_->has_graph(); }

 private:
  vsg::DeviceIndex* index_ = nullptr;
  py::ssize_t n_ = 0;
  py::ssize_t stride_ = 0;
};

}  // namespace

PYBIND11_MODULE(_vecstore_gpu, m) {
  m.doc() = "hand-written cuda kernels for vecstore (see gpu/SPEC.md)";

  // kernel capacities, so callers can size work without hardcoding
  m.attr("MAX_K") = vsg::max_k();
  m.attr("MAX_EF") = vsg::max_ef();

  m.def(
      "hello_add",
      [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
         const py::array_t<float, py::array::c_style | py::array::forcecast>&
             b) {
        if (a.ndim() != 1 || b.ndim() != 1) {
          throw py::value_error("hello_add takes 1-d arrays");
        }
        if (a.shape(0) != b.shape(0)) {
          throw py::value_error("hello_add operands must be the same length");
        }
        py::ssize_t n = a.shape(0);
        std::vector<py::ssize_t> shape;
        shape.push_back(n);
        py::array_t<float> out(shape);
        const float* pa = a.data();
        const float* pb = b.data();
        float* po = out.mutable_data();
        {
          py::gil_scoped_release release;
          vsg::hello_add(pa, pb, po, n);
        }
        return out;
      },
      py::arg("a"), py::arg("b"),
      "device round-trip check: elementwise a + b");

  m.def(
      "topk",
      [](const py::array& dmat, int k) {
        py::ssize_t nq = 0, n = 0;
        const float* pd = float_2d(dmat, "dmat", nq, n);
        if (nq < 1 || n < 1) throw py::value_error("dmat must be non-empty");
        check_k(k, n);
        std::vector<py::ssize_t> shape;
        shape.push_back(nq);
        shape.push_back((py::ssize_t)k);
        py::array_t<int32_t> ids(shape);
        py::array_t<float> dists(shape);
        int32_t* oi = ids.mutable_data();
        float* od = dists.mutable_data();
        {
          py::gil_scoped_release release;
          vsg::topk(pd, nq, n, k, oi, od, nullptr);
        }
        return py::make_tuple(ids, dists);
      },
      py::arg("dmat"), py::arg("k"),
      "k smallest per row of a (nq, n) fp32 matrix, closest first");

  py::class_<PyDeviceIndex>(m, "DeviceIndex")
      .def(py::init<const py::array&, py::ssize_t, const std::string&>(),
           py::arg("vectors"), py::arg("dim"), py::arg("metric") = "l2",
           "vectors is (n, stride) float16 with zero-padded rows")
      .def("set_graph", &PyDeviceIndex::set_graph, py::arg("adjacency"),
           py::arg("degrees"), py::arg("entry"))
      .def("distances", &PyDeviceIndex::distances, py::arg("queries"),
           py::arg("use_cublas") = false)
      .def(
          "distances_cublas",
          [](PyDeviceIndex& self, const py::array& queries) {
            return self.distances(queries, true);
          },
          py::arg("queries"))
      .def("brute_force", &PyDeviceIndex::brute_force, py::arg("queries"),
           py::arg("k") = 10, py::arg("use_cublas") = false)
      .def("hnsw", &PyDeviceIndex::hnsw, py::arg("queries"), py::arg("entries"),
           py::arg("k") = 10, py::arg("ef") = 50)
      .def("last_kernel_ms", &PyDeviceIndex::last_kernel_ms)
      .def("__len__", &PyDeviceIndex::size)
      .def_property_readonly("dim", &PyDeviceIndex::dim)
      .def_property_readonly("stride", &PyDeviceIndex::stride)
      .def_property_readonly("has_graph", &PyDeviceIndex::has_graph);
}
