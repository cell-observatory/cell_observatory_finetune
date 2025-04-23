// https://github.com/TimothyZero/MedVision/blob/f89d6cdc7fe9fda72d9b43521d7d402923afc10e/medvision/csrc/cuda/cuda_helpers.h
//
// Apache License
// Version 2.0, January 2004
// http://www.apache.org/licenses/

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>


#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

using at::Half;
using at::Tensor;
using phalf = at::Half;

const int CUDA_NUM_THREADS = 512;
const int THREADS_PER_BLOCK = CUDA_NUM_THREADS;

inline int GET_BLOCKS(const int N)
{
    int optimal_block_num = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}