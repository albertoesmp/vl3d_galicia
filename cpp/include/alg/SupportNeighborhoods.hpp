#ifndef VL3DPP_SUPPORT_NEIGHBORHOODS_
#define VL3DPP_SUPPORT_NEIGHBORHOODS_

// ***   INCLUDES   *** //
// ******************** //
#include <adt/kdtree/KDTree.hpp>
#include <alg/GridMesher.hpp>
#include <alg/FurthestPointSubsampler.hpp>
#include <util/VL3DPPMacros.hpp>
#include <util/VL3DPPException.hpp>
#include <util/TimeWatcher.hpp>
#include <util/logging/GlobalLogger.hpp>

#include <armadillo>

#include <string>
#include <vector>
#include <thread>
#include <sstream>

using vl3dpp::adt::kdtree::KDTree;
using vl3dpp::util::TimeWatcher;

namespace vl3dpp::alg{


// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief Support neighborhoods class.
 *
 * The support neighborhoods are a point cloud derived from another point cloud.
 * They often (but not necessarily) have less points than the input point cloud.
 * Support points are often used as neighborhood centers to carry out an
 * analysis in the point cloud without explicitly computing all the points.
 * They are also typically used to generate the input neighborhoods representing
 * a point cloud that will be fed into a neural network.
 *
 * @tparam XDecimalType The type of decimal number used to represent the
 *  structure space.
 * @tparam LabelType The type of integer number used to represent the class
 *  labels.
 * @tparam IndexType The type of integer number used to represent the indices.
 *  Indices are typically use to encode neighborhoods.
 */
template <typename XDecimalType, typename LabelType, typename IndexType>
class SupportNeighborhoods{
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief The neighborhood type. It can be either:
     *
     * <ul>
     * <li>cylinder</li>
     * <li>bounded_cylinder</li>
     * <li>sphere</li>
     * <li>rectangular2d</li>
     * <li>rectangular3d</li>
     * <li>knn2d</li>
     * <li>knn3d</li>
     * <li>bounded_knn2d</li>
     * <li>bounded_knn3d</li>
     * </ul>
     */
    std::string nbhType;
    /**
     * @brief The number of \f$K\f$-nearest neighbor for KNN- based
     *  neighborhoods.
     */
    IndexType nbhK;
    /**
     * @brief The axis-wise radii for neighborhoods that require at least one
     *  radius parameter.
     */
    arma::Col<XDecimalType> nbhRadii;
    /**
     * @brief The separation factor to distribute the support points such that
     *  they respect a minimum distance between them.
     *
     * It governs how many times the radius separates the support points. For
     * a given separation factor \f$k\f$, it will be that:
     *
     * \f[
     *  k \leq \frac{2}{\sqrt{n_x}}
     * \f]
     *
     * Where \f$n_x\f$ is the dimensionality of the structure space.
     *
     * <b>NOTE</b> that the separation factor is not always used. For example,
     * it has no implications when using a "fps" strategy. To the contrary,
     * it will have implications when using the "grid" strategy.
     *
     * @see SupportNeighborhoods::strategy
     */
    XDecimalType nbhSeparationFactor;
    /**
     * @brief The strategy to compute the support points. Valid strategies are:
     *
     * <ul>
     * <li>grid</li>
     * <li>fps</li>
     * <li>training_class_distribution</li>
     * </ul>
     */
    std::string strategy;
    /**
     * @brief The number of points to consider when using a strategy that
     *  targets a particular number of points (e.g., "fps").
     * @see SupportNeighborhoods::strategy
     */
    IndexType numPoints;
    /**
     * @brief Whether to use a fast approximation, when available.
     *
     * For a "fps" strategy it works as commented in
     * vl3dpp::rfield::DLFPSPreProcessor::fast.
     *
     * @see SupportNeighborhoods::strategy
     */
    short fast;
    /**
     * @brief How many samples per class such that [i] gives the number of
     *  samples for class \f$i\f$.
     */
    arma::Col<IndexType> trainingClassDistribution;
    /**
     * @brief Whether to force the support points to match a point in the input
     *  point cloud (true) or not (false).
     */
    bool centerOnPcloud;
    /**
     * @see vl3dpp::alg::GridMesher::extraNodes
     */
    bool extraNodes;
    /**
     * @brief How many threads must be used.
     *
     * If 0, then the number of threads depends on the current configuration.
     * If -1, then all available threads will be used.
     */
    int nthreads;

public:
    // ***  ATTRIBUTES : STATE  *** //
    // **************************** //
    /**
     * @brief Method that must be used to compute the neighborhood of a
     *  support point (xsupi) in the structure space (X) and store the indices
     *  of the neighbors from the structure space (X) in the vector of indices
     *  (Ii). The given KDTree (kdt) will be used to speedup neighborhood
     *  computations.
     */
    void (
        SupportNeighborhoods<
            XDecimalType, LabelType, IndexType
        >::*neighborhoodMethod
    )(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );

    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Instantiate a SupportNeighborhoods algorithm.
     * @see SupportNeighborhoods::nbhType
     * @see SupportNeighborhoods::nbhRadii
     * @see SupportNeighborhoods::nbhSeparationFactor
     * @see SupportNeighborhoods::strategy
     * @see SupportNeighborhoods::numPoints
     * @see SupportNeighborhoods::fast
     * @see SupportNeighborhoods::trainingClassDistribution
     * @see SupportNeighborhoods::centerOnPcloud
     * @see SupportNeighborhoods::nthreads
     */
    SupportNeighborhoods(
        std::string const &nbhType,
        IndexType const nbhK,
        arma::Col<XDecimalType> const &nbhRadii,
        XDecimalType const nbhSeparationFactor,
        std::string const &strategy,
        IndexType const numPoints,
        short const fast,
        arma::Col<IndexType> const &trainingClassDistribution,
        bool const centerOnPcloud,
        bool const extraNodes,
        int const nthreads
    );
    virtual ~SupportNeighborhoods() = default;

    // ***  MAIN METHODS  *** //
    // ********************** //
    /**
     * @brief Compute the requested support neighborhoods.
     * @param[in] X The input structure space
     *  \f$\pmb{X} \in \mathbb{R}^{m \times n_x}\f$.
     * @param[in] y The vector of point-wise labels (i.e., classes)
     *  \f$\pmb{y} \in \mathbb{Z}^{m}\f$.
     * @param[out] Xsup Matrix where the support points must be stored.
     * @param[out] I Vector of vectors (each internal vector with a potentially
     *  different dimensionality) with the indices of the neighbors in
     *  \f$\pmb{X}\f$ for each support neighborhood.
     */
    void computeAll(
        arma::Mat<XDecimalType> const &X,
        arma::Col<LabelType> const &y,
        arma::Mat<XDecimalType> &Xsup,
        std::vector<arma::Col<IndexType>> &I
    );
    /**
     * @brief Compute the support points that will be used as centers for the
     *  neighborhoods.
     * @param[in] X The input structure space
     *  \f$\pmb{X} \in \mathbb{R}^{m \times n_x}\f$.
     * @param[in] y The vector of point-wise labels (i.e., classes)
     *  \f$\pmb{y} \in \mathbb{Z}^{m}\f$.
     * @param[out] Xsup Matrix where the support points must be stored.
     */
    void computeSupportPoints(
        arma::Mat<XDecimalType> const &X,
        arma::Col<LabelType> const &y,
        arma::Mat<XDecimalType> &Xsup
    );


    // ***  SUPPORT METHODS  *** //
    // ************************* //
    /**
     * @brief Compute support points along the axis of a regular grid.
     * @see vl3dpp::alg::GridMesher
     */
    void computeGridSupport(
        arma::Mat<XDecimalType> const &X,
        arma::Mat<XDecimalType> &Xsup
    );
    /**
     * @brief Compute support points through furthest point subsampling (FPS).
     * @see vl3dpp::alg::FurthestPointSubsampler
     */
    void computeFPSSupport(
        arma::Mat<XDecimalType> const &X,
        arma::Mat<XDecimalType> &Xsup
    );
    /**
     * @brief Compute support points by selecting as many points as requested
     *  for each class.
     * @see SupportNeighborhoods::trainingClassDistribution
     */
    void computeTrainingClassDistributionSupport(
        arma::Mat<XDecimalType> const &X,
        arma::Col<LabelType> const &y,
        arma::Mat<XDecimalType> &Xsup
    );

    // ***  NEIGHBORHOOD METHODS  *** //
    // ****************************** //
    /**
     * @brief Wrapper to call SupportNeighborhoods::neighborhoodMethod
     * @see SupportNeighborhoods::neighborhoodMethod
     */
    inline void computeNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    ) {(this->*neighborhoodMethod)(X, xsupi, kdt, Ii);}
    /**
     * @brief Compute the spherical neighborhoods in the original structure
     *  space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findSphere
     */
    void computeSphericalNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );
    /**
     * @brief Compute the cylindrical neighborhoods in the original structure
     *  space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findCylinder
     */
    void computeCylindricalNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );
    /**
     * @brief Compute the bounded cylindrical neighborhoods in the original
     *  structure space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findBoundedCylinder
     */
    void computeBoundedCylindricalNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );
    /**
     * @brief Compute the 2D rectangular neighborhoods in the original
     *  structure space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findRectangle
     */
    void computeRectangular2DNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );
    /**
     * @brief Compute the 3D rectangular neighborhoods in the original
     *  structure space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findBoundedRectangle
     */
    void computeRectangular3DNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );
    /**
     * @brief Compute the 2D K-nearest neighbors neighborhoods in the original
     *  structure space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findKnn2D
     */
    void computeKnn2DNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );
    /**
     * @brief Compute the 3D K-nearest neighbors neighborhoods in the original
     *  structure space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findKnn
     */
    void computeKnn3DNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );
    /**
     * @brief Compute the bounded 2D K-nearest neighbors neighborhoods in the
     *  original structure space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findBoundedKnn2D
     */
    void computeBoundedKnn2DNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );
    /**
     * @brief Compute the bounded 3D K-nearest neighbors neighborhoods in the
     *  original structure space (X) centered on the support points.
     * @see vl3dpp::adt::kdtree::KDTree::findBoundedKnn3D
     */
    void computeBoundedKnn3DNeighborhood(
        arma::Mat<XDecimalType> const &X,
        arma::Col<XDecimalType> const &xsupi,
        KDTree<IndexType, XDecimalType> &kdt,
        arma::Col<IndexType> &Ii
    );

    // ***   UTIL METHODS   *** //
    // ************************ //
    /**
     * @brief Build a KDTree on the given structure space.
     * @param X The structure space on top of which the KDTree must be built.
     * @return Built KDTree.
     * @see vl3dpp::adt::kdtree::KDTree
     */
    KDTree<IndexType, XDecimalType> buildKDTree(
        arma::Mat<XDecimalType> const &X
    );

    /**
     * @brief Center the support points in the point cloud, i.e., any point
     *  in Xsup will be replaced by its closest neighbor in X. If any point
     *  is further than the radius parameter (typically radii[0]), then it will
     *  be discarded.
     * @see SupportNeighborhoods::computeSupportPoints
     * @see SupportNeighborhoods::computeGridSupport
     */
    void centerSupport(
        arma::Mat<XDecimalType> const &X,
        arma::Mat<XDecimalType> &Xsup
    );

public:
    // ***  GETTERs and SETTERs  *** //
    // ***************************** //
    /**
     * @see SupportNeighborhoods::numPoints
     */
    IndexType getNumPoints() const {return numPoints;}

};


#include <alg/SupportNeighborhoods.tpp>


}


#endif
