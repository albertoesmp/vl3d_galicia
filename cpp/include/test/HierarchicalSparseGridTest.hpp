#pragma once

#include <test/BaseTest.hpp>
#include <adt/grid/HierarchicalSparseGrid.hpp>

using namespace vl3dpp::adt::grid;

namespace vl3dpp::test{

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief HierarchicalSparseGridTest class.
 *
 * Class implementing test for the HierarchicalSparseGrid advanced data
 *  structure / abstract data type.
 */
class HierarchicalSparseGridTest : public BaseTest{
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
     * @brief Build the test for the hierarchical sparse grid data structure
     * @param eps The error tolerance threshold.
     */
    HierarchicalSparseGridTest(float const eps=1e-5) :
        BaseTest("Hierarchical sparse grid test"),
        eps(eps)
    {};
    virtual ~HierarchicalSparseGridTest() = default;

    // ***   R U N   *** //
    // ***************** //
    /**
     * @brief Test the correctness of vl3dpp::adt::grid::HierarchicalSparseGrid
     *
     * @return True if the test was successfully passed, false otherwise.
     * @see vl3dpp::test::BaseTest::run
     */
    bool run() override;

    // ***   TESTS   *** //
    // ***************** //
    /**
     * @brief Do a simple test for a 2D dense grid build as a sparse 2D grid.
     * @return True if the test was passed, false otherwise.
     */
    bool simple2DTestA();
    /**
     * @brief Do a simple test for a 2D sparse grid.
     * @return True if the test was passed, false otherwise.
     */
    bool simple2DTestB();
    /**
     * @see HierarchicalSparseGridTest::simple2DTestB
     */
    bool simple2DTestC();
    /**
     * @see HierarchicalSparseGridTest::simple2DTestB
     */
    bool simple2DTestD();
    /**
     * @see HierarchicalSparseGridTest::simple2DTestB
     */
    bool simple2DTestE();
    /**
     * @see HierarchicalSparseGridTest::simple2DTestB
     */
    bool simple2DTestF();
    /**
     * @see HierarchicalSparseGridTest::simple2DTestB
     */
    bool simple2DTestG();
    /**
     * @see HierarchicalSparseGridTest::simple2DTestB
     */
    bool simple2DTestH();
    /**
     * @brief Reordered X version of simple2DTestB.
     * @see HierarchicalSparseGridTest::simple2DTestB
     */
    bool simple2DTestI();
    /**
     * @brief Reordered X version of simple2DTestH.
     * @see HierarchicalSparseGridTest::simple2DTestH
     */
    bool simple2DTestJ();
    /**
     * @see HierarchicalSparseGridTest::simple2DTestB
     */
    bool simple2DTestK();
    /**
     * @brief Do a simple test for a 3D sparse grid.
     * @return True if the test was passed, false otherwise.
     */
    bool simple3DTestA();

protected:
    // ***  VALIDATION METHODS  *** //
    // **************************** //
    /**
     * @brief Check whether the given submanifold map matches the expected
     *  reference submanifold map or not.
     * @tparam IndexType The type of index defining the submanifold map.
     * @param hRef The reference submanifold map.
     * @param h The given submanifold map to be validated.
     * @return True if the given submanifold map is valid (i.e., matches its
     *  reference), false otherwise.
     */
    template <typename IndexType>
    bool validateSubmanifoldMap(
        std::map<IndexType, IndexType> const &hRef,
        std::map<IndexType, IndexType> const &h
    ) const;
    /**
     * @brief Check whether the given downsampling map matches the expected
     *  reference downsampling map or not.
     * @tparam IndexType The type of index defining the downsampling map.
     * @param hDref The reference downsampling map.
     * @param hD The given downsampling map to be validated.
     * @return True if the given downsampling map is valid (i.e., matches its
     *  reference), false otherwise.
     */
    template <typename IndexType>
    bool validateDownsamplingMap(
        std::vector<IndexType> const &hDref,
        std::vector<IndexType> const &hD
    ) const;
    /**
     * @brief Check whether the given upsampling map matches the expected
     *  reference upsampling map or not.
     * @tparam IndexType The type of index defining the upsampling map.
     * @param hUref The reference upsampling map.
     * @param hU The given upsampling map to be validated.
     * @return True if the given upsampling map is valid (i.e., matches its
     *  reference), false otherwise.
     */
    template <typename IndexType>
    bool validateUpsamplingMap(
        std::vector<IndexType> const &hUref,
        std::vector<IndexType> const &hU
    ) const;
};

}