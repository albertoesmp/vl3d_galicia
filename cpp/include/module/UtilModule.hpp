/**
 * Provides functions wrapping VL3DPP to be easily called from the VL3D python
 * software.
 * More concretely, the functions here wrap utils (e.g., the C++ logging
 * system).
 */

// ***   INCLUDES   *** //
// ******************** //
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <carma>
#include <armadillo>

#include <string>
#include <vector>

namespace py = pybind11;

namespace vl3dpp::pymods{

void logging_enable(){
    LOGGER->fullLogging();
}

void logging_disable(){
    LOGGER->silence();
}


}
