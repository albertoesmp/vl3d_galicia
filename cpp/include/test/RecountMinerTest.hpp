#pragma once

#include <test/BaseTest.hpp>
#include <mining/RecountMiner.hpp>

using namespace vl3dpp::mining;

namespace vl3dpp::test{

class RecountMinerTest : public BaseTest{
protected:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief Error tolerance.
     */
    float eps;

    /**
     * @brief The 3D structure space matrix
     *  \f$\pmb{X} \in \mathbb{R}^{m \times 3}\f$ representing the test point
     *  cloud.
     */
    arma::Mat<float> X;
    /**
     * @brief The input feature space matrix that will be smoothed during
     *  the test.
     */
    arma::Mat<float> F;
    /**
     * @brief The feature space matrix
     *  \f$\pmb{F} \in \mathbb{R}^{m \times n_f}\f$ representing the test point
     *  cloud.
     *
     * It contains the reference features to compare against the ones computed
     * during the test.
     */
    arma::Mat<float> Fref;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build the test for the RecountMiner.
     * @see vl3dpp::mining::RecountMiner
     * @see RecountMinerTest::eps
     */
    RecountMinerTest(float const eps=1e-3);
    virtual ~RecountMinerTest() = default;

    // ***   R U N   *** //
    // ***************** //
    /**
     * @brief Test that RecountMiner yields the expected results.
     *
     * @return True if the test was successfully passed, false otherwise.
     * @see vl3dpp::test::BaseTest::run
     */
    bool run() override;
};

}