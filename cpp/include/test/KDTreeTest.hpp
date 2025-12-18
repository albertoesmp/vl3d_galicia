#pragma once

#include <test/BaseTest.hpp>
#include <adt/kdtree/KDTree.hpp>
#include <armadillo>
#include <memory>

using namespace vl3dpp::adt::kdtree;

namespace vl3dpp::test{

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief KDTreeTest class
 *
 * Class implementing test for the KDTree advanced data structure / abstract
 * data type.
 */
class KDTreeTest : public BaseTest{
protected:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief Error tolerance.
     */
    double eps;
    /**
     * @brief The 3D structure space matrix
     *  \f$\pmb{X} \in \mathbb{R}^{m \times 3}\f$ representing the test point
     *  cloud.
     */
    arma::Mat<double> X;
    /**
     * @brief The support points, i.e., those points whose neighborhoods must
     *  be found.
     */
    arma::Mat<double> Xsup;
    /**
     * @brief The KDTree to be used for the tests.
     * @see vl3dpp::adt::kdtree::KDTree
     */
    std::shared_ptr<KDTree<size_t, double>> kdt;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build the test for the KDTree.
     * @see vl3dpp::adt::kdtree::KDTree
     * @see KDTreeTest::eps
     */
    KDTreeTest(double const eps=1e-5);
    virtual ~KDTreeTest() = default;

    // ***   R U N   *** //
    // ***************** //
    /**
     * @brief Test that the KDTree yields the expected results.
     *
     * @return True if the test was successfully passed, false otherwise.
     * @see vl3dpp::test::BaseTest::run
     */
    bool run() override;

    // ***  UTIL METHODS  *** //
    // ********************** //
    /**
     * @brief Validate that the given neighborhoods match those in the file
     *  at the given path.
     *
     * @param N The neighborhoods to be tested.
     * @param NPath Path to the file containing the reference neighborhood data
     *  to validate against.
     * @param force2D Whether to force 2D dimensionality on the points (true)
     *  or not (false).
     * @param sort Whether to sort the indices before the checks (true) or not
     *  (false).
     * @return True if the given neighborhoods are valid, false otherwise.
     */
    bool validateNeighborhoods(
        arma::Mat<size_t> const &N,
        std::string const &NPath,
        bool const force2D=false,
        bool const sort=false
    );
    /**
     * @brief Validate that the given neighborhoods match those in the file
     *  at the given path.
     *
     * @param N The neighborhoods to be tested.
     * @param NPath Path to the file containing the reference neighborhood data
     *  to validate against.
     * @param force2D Whether to force 2D dimensionality on the points (true)
     *  or not (false).
     * @param bounded  Whether to apply a correction on the boundaries of
     *  the neighborhoods (true) or not (false).
     * @param sort Whether to sort the indices before the checks (true) or not
     *  (false).
     * @return True if the given neighborhoods are valid, false otherwise.
     */
    bool validateNeighborhoods(
        std::vector<arma::Col<size_t>> const &N,
        std::string const &NPath,
        bool const force2D=false,
        bool const bounded=false,
        bool const sort=false
    );
    /**
     * @brief Validate that the point-wise distances between each neighbor
     *  and the center point of the neighborhood are correct.
     * @param D The distances to be validated.
     * @param DPath Path to the file with the reference distances to validate
     *  against them.
     * @return True if the given distances are valid, false otherwise.
     */
    bool validateDistances(
        arma::Mat<double> const &D,
        std::string const &DPath
    );
    /**
     * @brief Validate that the point-wise distances between each neighbor
     *  and the center point of the neighborhood are correct.
     * @param D The distances to be validated.
     * @param DPath Path to the file with the reference distances to validate
     *  against them.
     * @param bounded  Whether to apply a correction on the boundaries of
     *  the neighborhoods (true) or not (false).
     * @return True if the given distances are valid, false otherwise.
     */
    bool validateDistances(
        std::vector<arma::Col<double>> const &D,
        std::string const &DPath,
        bool const bounded=false
    );
};

};