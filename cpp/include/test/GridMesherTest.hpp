#pragma once

#include <test/BaseTest.hpp>
#include <alg/GridMesher.hpp>

using namespace vl3dpp::alg;

namespace vl3dpp::test{


class GridMesherTest : public BaseTest{
protected:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief Error tolerance.
     */
    float eps;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build the test for the grid mesher algorithm.
     * @see vl3dpp::alg
     */
    GridMesherTest(float const eps=1e-5) :
        BaseTest("Grid mesher test"),
        eps(eps)
    {};
    virtual ~GridMesherTest() = default;

    // ***   R U N   *** //
    // ***************** //
    /**
     * @brief Test the correctness of vl3dpp::alg::GridMesher
     *
     * @return True if the test was successfully passed, false otherwise.
     * @see vl3dpp::test::BaseTest::run
     */
    bool run() override;

    // ***  UTIL METHODS  *** //
    // ********************** //
    /**
     * @brief Check whether the given grids are equal or not.
     * @return True if given grids are equal, false otherwise.
     */
    template <typename DecimalType>
    bool checkGridEquality(
        arma::Mat<DecimalType> const &Ga,
        arma::Mat<DecimalType> const &Gb
    ) const;
};


}