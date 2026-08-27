// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

#include "common.hpp"
#include <cstdlib>

// Cap allocations to the wasm32 address space to reject bogus/huge sizes.
static constexpr size_t kMaxAllocationSize = static_cast<size_t>(1) << 32;

extern "C" {
APIFUNC void *memory_allocate(size_t size) {
  if (size == 0 || size > kMaxAllocationSize) {
    return nullptr;
  }
  return malloc(size);
}
APIFUNC void memory_free(void *ptr) {
  if (ptr != nullptr) {
    free(ptr);
  }
}
}