// Build compatibility for the pinned author's EXPANS_ALLOC error path only.
// The author files remain unchanged. Successful allocation / codec paths do
// not call this overload. Forward unique_ptr errors to the existing handler.
#pragma once
#include <memory>
#include <cuda_runtime.h>
struct openlake_expans_codec;
namespace {
int fail_cuda(openlake_expans_codec*, cudaError_t, const char*);
inline int fail_cuda(const std::unique_ptr<openlake_expans_codec>& codec,
                     cudaError_t status, const char* operation) {
  return fail_cuda(codec.get(), status, operation);
}
}
