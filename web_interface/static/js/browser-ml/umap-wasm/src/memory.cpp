// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

#include "common.hpp"

extern "C" {
APIFUNC void *memory_allocate(size_t size) { return malloc(size); }
APIFUNC void memory_free(void *ptr) { free(ptr); }
}