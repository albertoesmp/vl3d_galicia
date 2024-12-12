// ***   INCLUDES   *** //
// ******************** //
#include <test/BaseTest.hpp>
#include <iomanip>

using namespace vl3dpp::test;

// ***   T E S T   *** //
// ******************* //
/**
 * @brief Do the test and report its status (whether passed or not)
 *  through the given output stream.
 * @param out The output stream to report the test status.
 * @param color Whether to include colors in the output stream (true) or
 *  not (false).
 * @return True if test was successfully passed, false otherwise.
 * @see BaseTest::run
 */
bool BaseTest::operator() (std::ostream &out, bool const color){
    // Run the test
    bool const status = correctlyConstructed && run();

    // Report test status
    if(color) out << "\033[1m";
    out << "TEST ";
    if(color) out << "\033[0m";
    out << std::setw(52) << std::left << name.c_str() << " ";
    if(color) out << "\033[1m";
    out << "[";
    if(color){
        if(status) out << "\033[32m";
        else out << "\033[31m";
    }
    out << (status ? "PASSED" : "FAILED");
    if(color) out << "\033[0m\033[1m";
    out << "]";
    if(color) out << "\033[0m";
    out << std::endl;

    // Return status
    return status;
}
