#pragma once

#include <test/BaseTest.hpp>
#include <rfield/DLFPSPostProcessor.hpp>

using namespace vl3dpp::rfield;

namespace vl3dpp::test{

class ParallelReceptiveFieldTest : public BaseTest{
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
     * @brief Build the test for the parallel execution of receptive
     *  field-related computations.
     * @see vl3dpp::rfield
     */
    ParallelReceptiveFieldTest(float const eps=1e-3) :
        BaseTest("Parallel receptive field test"),
        eps(eps)
    {};
    virtual ~ParallelReceptiveFieldTest() = default;

    // ***   R U N   *** //
    // ***************** //
    /**
     * @brief Test that the receptive fields can be run in parallel.
     *
     * @return True if the test was successfully passed (i.e., the parallel
     *  executions finish without concurrency issues).
     * @see vl3dpp::test::BaseTest::run
     */
    bool run() override;

    // ***  TEST METHODS  *** //
    // ********************** //
    /**
     * @brief Check that post-processing logic of receptive fields can be
     *  successfully run in parallel, i.e., they run without failures.
     * @return True if the post-processing runs in parallel with no failures,
     *  false otherwise.
     */
    bool doPostProcessingTests();
};


}