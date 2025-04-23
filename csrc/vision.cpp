// https://github.com/TimothyZero/MedVision/blob/f89d6cdc7fe9fda72d9b43521d7d402923afc10e/medvision/csrc/vision.cpp#L29
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


#include "nms_3d.h"
#include "roi_align_3d.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // operators
    m.def("nms_3d", &nms_3d, "non-maximum suppression for 3d");
    
    m.def("roi_align_3d_forward", &roi_align_3d_forward, "roi_align_3d_forward");
    m.def("roi_align_3d_backward", &roi_align_3d_backward, "roi_align_3d_backward");

}