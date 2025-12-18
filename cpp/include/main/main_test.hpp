#pragma once

// ***   INCLUDES   *** //
// ******************** //
// Test suites
#include <test/TestSuite.hpp>

// ADT tests
#include <test/OctreeTest.hpp>
#include <test/KDTreeTest.hpp>
#include <test/LazySupportGridTest.hpp>
#include <test/SparseGridTest.hpp>
#include <test/HierarchicalSparseGridTest.hpp>

// Algorithm tests
#include <test/GridMesherTest.hpp>

// Receptive fields tests
#include <test/ParallelReceptiveFieldTest.hpp>

// Data mining tests
#include <test/HeightFeatsMinerTest.hpp>
#include <test/SmoothFeatsMinerTest.hpp>
#include <test/RecountMinerTest.hpp>

// Logging system
#include <util/logging/BasicLogger.hpp>
#include <util/logging/GlobalLogger.hpp>

#include <cstdlib>
#include <iostream>


// Determine whether to use color or not depending on the OS
#if defined(_WIN32) || defined(_WIN64)
#define TEST_COLOR false
#else
#define TEST_COLOR true
#endif


namespace vl3dpp::test{


// ***   T E S T S   *** //
// ********************* //
int run_adt_tests(){
    // Build test suite
    TestSuite ts("Advanced data types");
    ts.addTest(std::make_shared<OctreeTest>());
    ts.addTest(std::make_shared<KDTreeTest>());
    ts.addTest(std::make_shared<LazySupportGridTest>());
    ts.addTest(std::make_shared<SparseGridTest>());
    ts.addTest(std::make_shared<HierarchicalSparseGridTest>());

    // Run test suite
    ts(std::cout, TEST_COLOR);

    // Return number of failed tests
    return ts.getFailedCount();
}

int run_alg_tests(){
    // Build test suite
    TestSuite ts("Algorithms");
    ts.addTest(std::make_shared<GridMesherTest>());

    // Run test suite
    ts(std::cout, TEST_COLOR);

    // Return number of failed tests
    return ts.getFailedCount();
}

int run_rfield_tests(){
    // Build test suite
    TestSuite ts("Receptive fields");
    ts.addTest(std::make_shared<ParallelReceptiveFieldTest>());

    // Run test suite
    ts(std::cout, TEST_COLOR);

    // Return number of failed tests
    return ts.getFailedCount();
}

int run_mining_tests(){
    // Build test suite
    TestSuite ts("Data mining");
    ts.addTest(std::make_shared<HeightFeatsMinerTest>());
    ts.addTest(std::make_shared<SmoothFeatsMinerTest>());
    ts.addTest(std::make_shared<RecountMinerTest>());

    // Run test suite
    ts(std::cout, TEST_COLOR);

    // Return number of failed tests
    return ts.getFailedCount();
}


// ***   RUN THE TESTS   *** //
// ************************* //
/**
 * @brief Run the tests.
 * @return The number of failed tests.
 */
int main_test(){
    // Disable logging
    LOGGER->silence();

    // Run tests
    int failedCount = 0;
    failedCount += run_adt_tests();
    failedCount += run_alg_tests();
    failedCount += run_rfield_tests();
    failedCount += run_mining_tests();

    // Return number of failed tests
    return failedCount;
}


}