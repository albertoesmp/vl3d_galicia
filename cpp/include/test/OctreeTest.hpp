#pragma once

#include <test/BaseTest.hpp>
#include <adt/octree/Octree.hpp>
#include <armadillo>
#include <memory>

using namespace vl3dpp::adt::octree;

namespace vl3dpp::test{

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief OctreeTest class
 *
 * Class implementing tests for the Octree advanced data structure / abstract
 * data type.
 */
class OctreeTest : public BaseTest {
protected:
    // ***  ATTRIBUTES  *** //
    // ******************** //
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
     * @brief The Octree to be used for the tests.
     * @see vl3dpp::adt::octree::Octree
     */
    std::shared_ptr<Octree<size_t, double>> octree;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build the test for the octree.
     * @see OctreeTest::eps
     */
    OctreeTest(double const eps=1e-5);
    virtual ~OctreeTest() = default;

    // ***  R U N  *** //
    // *************** //
    /**
     * @brief Test that the Octree yields the expected results.
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
     * @param X The structure space matrix representing the point cloud.
     * @param Xsup the structure space matrix representing the centers of the
     *  neighborhoods.
     * @param N The neighborhoods to be tested.
     * @param NPath Path to the file containing the reference neighborhood data
     *  to validate against.
     * @param force2D Whether to force 2D dimensionality on the points (true)
     *  or not (false).
     * @param sort Whether to sort the indices before the checks (true) or not
     *  (false).
     * @param eps The error tolerance threshold for operations involving
     *  decimal numbers.
     * @return True if the given neighborhoods are valid, false otherwise.
     */
    static bool _validateNeighborhoods(
        arma::Mat<double> const &X,
        arma::Mat<double> const &Xsup,
        arma::Mat<size_t> const &N,
        std::string const &NPath,
        bool const force2D=false,
        bool const sort=false,
        double const eps=1e-5
    );
    /**
     * @see OctreeTest::_validateNeighborhoods(arma::Mat<double> const &, arma::Mat<double> const &, arma::Mat<size_t> const &, std::string const &, bool const, bool const, double const)
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
     * @param X The structure space matrix representing the point cloud.
     * @param Xsup the structure space matrix representing the centers of the
     * @param N The neighborhoods to be tested.
     * @param NPath Path to the file containing the reference neighborhood data
     *  to validate against.
     * @param force2D Whether to force 2D dimensionality on the points (true)
     *  or not (false).
     * @param bounded  Whether to apply a correction on the boundaries of
     *  the neighborhoods (true) or not (false).
     * @param sort Whether to sort the indices before the checks (true) or not
     *  (false).
     * @param eps The error tolerance threshold for operations involving
     *  decimal numbers.
     * @return True if the given neighborhoods are valid, false otherwise.
     */
    static bool _validateNeighborhoods(
        arma::Mat<double> const &X,
        arma::Mat<double> const &Xsup,
        std::vector<arma::Col<size_t>> const &N,
        std::string const &NPath,
        bool const force2D=false,
        bool const bounded=false,
        bool const sort=false,
        double const eps=1e-5
    );
    /**
     * @see OctreeTest::_validateNeighborhoods(arma::Mat<double> const &, arma::Mat<double> const &, std::vector<arma::Col<size_t>> const &, std::string const &, bool const, bool const, bool const, double const)
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
     * @param eps The error tolerance threshold for operations involving
     *  decimal numbers.
     * @return True if the given distances are valid, false otherwise.
     */
    static bool _validateDistances(
        arma::Mat<double> const &D,
        std::string const &DPath,
        double const eps
    );
    /**
     * @see OctreeTest::_validateDistances(arma::Mat<double> const &, std::string const &, double const)
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
     * @param eps The error tolerance threshold for operations involving
     *  decimal numbers.
     * @return True if the given distances are valid, false otherwise.
     */
    static bool _validateDistances(
        std::vector<arma::Col<double>> const &D,
        std::string const &DPath,
        bool const bounded,
        double const eps
    );
    /**
     * @see OctreeTest::_validateDistances(std::vector<arma::Col<double>> const &, std::string const &, double const)
     */
    bool validateDistances(
        std::vector<arma::Col<double>> const &D,
        std::string const &DPath,
        bool const bounded=false
    );
};
}