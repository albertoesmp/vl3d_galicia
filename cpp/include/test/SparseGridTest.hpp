#pragma once

#include <test/BaseTest.hpp>
#include <adt/grid/SparseGrid.hpp>

using namespace vl3dpp::adt::grid;

namespace vl3dpp::test{

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief SparseGridTest class.
 *
 * Class implementing test for the SparseGrid advanced data structure /
 *  abstract data type.
 */
class SparseGridTest : public BaseTest{
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
     * @brief Build the test for the sparse grid data structure.
     * @param eps The error tolerance threshold.
     */
    SparseGridTest(float const eps=1e-5) :
        BaseTest("Sparse grid test"),
        eps(eps)
    {};
    virtual ~SparseGridTest() = default;

    // ***   R U N   *** //
    // ***************** //
    /**
     * @brief Test the correctness of vl3dpp::adt::grid::SparseGrid
     *
     * @return True if the test was successfully passed, false otherwise.
     * @see vl3dpp::test::BaseTest::run
     */
    bool run() override;


};


}