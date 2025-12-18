#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <carma>
#include <armadillo>
#include <main/main_test.hpp>

#include <util/logging/BasicLogger.hpp>
#include <util/logging/GlobalLogger.hpp>

// Explicitly including the .cpp files is necessary to avoid PyBind11 segfault
#include <module/vl3dpp_algorithm.cpp>
#include <module/vl3dpp_data_mining.cpp>
#include <module/vl3dpp_rfield.cpp>

#include <string>

namespace py = pybind11;

// Initialize C++ logging system
std::shared_ptr<BasicLogger> LOGGER = make_shared<BasicLogger>("VL3DPP.log");
#include <module/UtilModule.hpp>

// ***  DANGLING FUNCTIONS  *** //
// **************************** //
std::string get_hello_world() {
    return "HELLO WORLD";
}

// ***  PYVL3DPP DECLARATIONS  *** //
// ******************************* //
void vl3dpp_data_mining_init(py::module &m);
void vl3dpp_rfield_init(py::module &m);
void vl3dpp_algorithm_init(py::module &m);

// ***   PYVL3DPP MODULE   *** //
// *************************** //
PYBIND11_MODULE(pyvl3dpp, m){
    // Hello world
    m.def(
        "get_hello_world",
        &get_hello_world,
        "Return the HELLO WORLD string"
    );
    // TODO Rethink : Move main_test to vl3dpp_test ?
    m.def(
        "main_test",
        &vl3dpp::test::main_test,
        "Return the number of failed tests"
    );
    // Logging system
    m.def(
        "logging_enable",
        &vl3dpp::pymods::logging_enable,
        "Enable C++ logging."
    );
    m.def(
        "logging_disable",
        &vl3dpp::pymods::logging_disable,
        "Disable C++ logging."
    );
    // Data mining module
    vl3dpp_data_mining_init(m);
    // Receptive field module
    vl3dpp_rfield_init(m);
    // Algorithm module
    vl3dpp_algorithm_init(m);
}