/**
 * @author Alberto M. Esmoris Pena
 * @brief Main entry point to run the tests of VL3D++.
 * @param argc The number of input arguments.
 * @param argv The array of input arguments as C strings.
 * @return The exit code.
 */

// ***   INCLUDES   *** //
// ******************** //
#include <main/main_test.hpp>


using namespace vl3dpp::test;


// ***  INITIALIZATION  *** //
// ************************ //
// Initialize C++ logging system
std::shared_ptr<BasicLogger> LOGGER = make_shared<BasicLogger>(
    "VL3DPP_test.log"
);


// ***   M A I N   *** //
// ******************* //
int main (int argc, char **argv){
    // Initialize C++ logging system
    std::shared_ptr<BasicLogger> LOGGER = make_shared<BasicLogger>(
        "VL3DPP_test.log"
    );

    // Before running tests
    std::cout << "Running VL3D++ tests ...\n" << std::endl;

    // Run tests
    int failedCount = main_test();

    // When all tests passed
    bool passed = failedCount == 0;
    if(passed) {
        std::cout << "All tests PASSED!  :)" << std::endl;
        return EXIT_SUCCESS;
    }
    // When at least one tests failed
    else{
        std::cout   << "Some tests (" << failedCount << ") "
                    << "FAILED!  :(" << std::endl;
        return EXIT_FAILURE;
    }
}