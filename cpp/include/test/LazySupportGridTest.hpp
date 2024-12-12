#pragma once

// ***   INCLUDES   *** //
// ******************** //
#include <test/BaseTest.hpp>
#include <adt/grid/LazySupportGrid.hpp>
#include <armadillo>
#include <memory>

using namespace vl3dpp::adt::grid;



namespace vl3dpp::test {

// ***   CLASS   *** //
// ***************** //
class LazySupportGridTest : public BaseTest{
protected:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief Error tolerance.
     */
    double eps;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build the test for the LazySupportGridTest.
     * @see LazySupportGridTest::eps
     */
    LazySupportGridTest(double const eps=1e-5);
    virtual ~LazySupportGridTest() = default;


    // ***   R U N   *** //
    // ***************** //
    /**
     * @brief Test that the LazySupportGrid yields the expected results.
     *
     * @return True if the test was successfully passed, false otherwise.
     * @see vl3dpp::test::BaseTest::run
     */
    bool run() override;


    // ***  UTIL METHODS  *** //
    // ********************** //
    /**
     * @brief Test that LazySupportGrid yields the expected results.
     *
     * @param grid The lazy support grid to be evaluated.
     * @param Y The matrix with the reference values that the lazy support grid
     *  must yield.
     * @return True if the test was successfully passed, false otherwise.
     */
    bool validateGrid(
        LazySupportGrid<double> const &grid,
        arma::Mat<double> const &Y
    );
};

}