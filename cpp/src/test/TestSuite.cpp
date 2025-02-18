// ***   INCLUDES   *** //
// ******************** //
#include <test/TestSuite.hpp>
#include <iostream>
#include <iomanip>

using namespace vl3dpp::test;

// ***   R U N   *** //
// ***************** //
bool TestSuite::run(){
    return _run(
        [&] (std::shared_ptr<BaseTest> bt) -> bool {
            return bt->run();
        }
    );
}

bool TestSuite::operator() (std::ostream &out, bool const color){
    // Report before running tests
    if(color) out << "\033[1m";
    std::cout << "TEST SUITE: ";
    if(color) out << "\033[0m";
    std::cout << name << "\n";

    // Run tests
    _run(
        [&] (std::shared_ptr<BaseTest> bt) -> bool {
            return (*bt)(out, color);
        }
    );

    // Report after running tests
    if(failedTests == 0) { // All tests passed
        if(color) out << "\033[1m\033[32m";
        std::cout << "All the " << tests.size() << " tests in the suite "
                  << "\"" << name << "\" were passed.";
    }
    else{ // At least one test failed
        if(color) out << "\033[1m\033[31m";
        std::cout   << "Unfortunately, " << failedTests << " test out of "
                    << tests.size() << " failed for the suite "
                    << name << ".";
    }
    if(color) out << "\033[0m";
    out << "\n" << std::endl;

    // Return
    return failedTests == 0;
}

bool TestSuite::_run(std::function<bool(std::shared_ptr<BaseTest>)> testCaller){
    failedTests = 0;
    for(std::shared_ptr<BaseTest> bt : tests){
        if(!testCaller(bt)) ++failedTests;
    }
    return failedTests == 0;
}
