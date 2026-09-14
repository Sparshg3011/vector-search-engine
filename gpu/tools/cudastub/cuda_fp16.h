#pragma once

// Structural stand-in for fp16 support - see cuda_runtime.h for what
// the stubs are and are not for.

#include "cuda_runtime.h"

struct __half {
  unsigned short bits;
};

// two halves in one load; permissive about how it is built and read
struct __half2 {
  __half lo, hi;
  __half2() : lo(), hi() {}
  template <class... A>
  __half2(A&&...) : lo(), hi() {}
  __half operator[](int i) const { return i ? hi : lo; }
};

float __half2float(__half h);
__half __float2half(float f);
template <class T>
float __half2float(const T&) { return 0.0f; }

// two halves -> two floats in one call
float2 __half22float2(const __half2& h);
float __low2float(const __half2& h);
float __high2float(const __half2& h);
