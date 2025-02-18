#ifndef VL3DPP_RECOUNT_MINER_
#define VL3DPP_RECOUNT_MINER_

// ***   INCLUDES   *** //
// ******************** //
#include <adt/kdtree/KDTree.hpp>
#include <util/TimeWatcher.hpp>
#include <util/logging/GlobalLogger.hpp>
#include <util/VL3DPPException.hpp>
#include <util/VL3DPPMacros.hpp>

#include <armadillo>
#include <omp.h>

#include <functional>
#include <vector>
#include <string>
#include <thread>


namespace vl3dpp::mining{

using vl3dpp::adt::kdtree::KDTree;

using std::function;
using std::vector;
using std::string;


// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing a recount-based data miner.
 *
 *
 * @tparam XDecimalType The data type for the decimal numbers in the context
 *  of the point cloud's structure space.
 * @tparam FDecimalType The data type for the decimal numbers in the context
 *  of the point cloud's feature space.
 */
template <typename XDecimalType, typename FDecimalType>
class RecountMiner {
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief String specifying the type of neighborhood. Supported
     *  neighborhoods are:
     *
     * <ul>
     *  <li>knn</li>
     *  <li>knn2d</li>
     *  <li>sphere</li>
     *  <li>cylinder</li>
     *  <li>rectangular3d</li>
     *  <li>rectangular2d</li>
     *  <li>boundedcylinder</li>
     * </ul>
     */
    string nbhType;
    /**
     * @brief The number of nearest neighbors to consider when using the
     * k-nearest neighbors neighborhood.
     */
    arma::uword nbhK;
    /**
     * @brief The neighborhood radius for spherical, cylindrical, and
     * rectangular neighborhoods.
     *
     * Note that for rectangular neighborhoods, the radius defines half
     * the length of a side.
     *
     * When using rectangular neighborhoods, the radius is considered as a
     * vector that defines the axis-wise radius. When using spherical or
     * cylindrical neighborhoods, only the first component of the vector is
     * considered as the radius.
     */
    arma::Col<XDecimalType> const nbhRadius;
    /**
     * @brief The lower bound \f$\tau_*\f$ for bounded cylindrical
     *  neighborhoods.
     *
     * For example, for a vertical cylinder, the neighborhood is centered
     * on the point \f$\pmb{x}_{i*} \in \mathbb{R}^{m \times 3}\f$. Any
     * \f$j\f$-th point in the neighborhood that satisfies
     * \f$z_j-z_i \geq \tau_*\f$ will be considered inside bounds, otherwise
     * outside.
     */
    XDecimalType const nbhLowerBound;
    /**
     * @brief The upper bound \f$\tau^*\f$ for bounded cylindrical
     *  neighborhoods.
     *
     * For example, for a vertical cylinder, the neighborhood is centered
     * on the point \f$\pmb{x}_{i*} \in \mathbb{R}^{m \times 3}\f$. Any
     * \f$j\f$-th point in the neighborhood that satisfies
     * \f$z_j-z_i \leq \tau^*\f$ will be considered inside bounds, otherwise
     * outside.
     */
    XDecimalType const nbhUpperBound;
    /**
     * @brief Vector of flags specifying whether the \f$i\f$-th filter must
     *  ignore NaNs (true) or not (false).
     */
    std::vector<bool> ignoreNan;
    /**
     * @brief Vector of flags specifying whether the \f$i\f$-th filter must
     *  compute the absolute frequencies (true) or not (false).
     */
    std::vector<bool> absFreq;
    /**
     * @brief Vector of flags specifying whether the \f$i\f$-th filter must
     *  compute the relative frequencies (true) or not (false).
     */
    std::vector<bool> relFreq;
    /**
     * @brief Vector of flags specifying whether the \f$i\f$-th filter must
     *  compute the surface density (true) or not (false).
     *
     * Note that when the surface density is computed for a neighborhood that
     * is not governed by a radius parameter (e.g., k-nearest neighbors), then
     * the radius is computed for each neighborhood depending on the distance
     * between the center point and its furthest point in the neighborhood.
     */
    std::vector<bool> surfDens;
    /**
     * @brief Vector of flags specifying whether the \f$i\f$-th filter must
     *  compute the volume density (true) or not (false).
     *
     * Note that when the volume density is computed for a neighborhood that
     * is not governed by a radius parameter (e.g., k-nearest neighbors), then
     * the radius is computed for each neighborhood depending on the distance
     * between the center point and its furthest point in the neighborhood.
     */
    std::vector<bool> volDens;
    /**
     * @brief Vector of integers specifying how many vertical segments must be
     *  computed by the \f$i\f$-th filter. Note that zero means no vertical
     *  segments will be computed. The vertical segments will be linearly
     *  spaced. Each vertical segment filter will return the number of
     *  vertical segments with at least one point (i.e., the count of
     *  non-empty vertical segments).
     */
    std::vector<int> vertSeg;
    /**
     * @brief Vector of integers specifying how many rings must be computed
     *  by the \f$i\f$-th filter. Note that zero means no rings will be
     *  computed. The rings will be linearly spaced. Each ring filter will
     *  return the number of rings with at least one point (i.e., the count of
     *  non-empty rings).
     *
     * Note that when the rings (annuli) are computed for a neighborhood that
     * is not governed by a radius parameter (e.g., k-nearest neighbors), then
     * the radius is computed for each neighborhood depending on the distance
     * between the center point and its furthest point in the neighborhood.
     */
    std::vector<int> rings;
    /**
     * @brief Vector of integers specifying how many radial boundaries must
     *  be computed by the \f$i\f$-th filter. Note that zero means no radial
     *  boundaries will be comptued. The radial boundaries will be linearly
     *  spaced. Each radial boundary filter will return the number of non-empty
     *  radial regions.
     *
     * Note that when the spherical shells are computed for a neighborhood that
     * is not governed by a radius parameter (e.g., k-nearest neighbors), then
     * the radius is computed for each neighborhood depending on the distance
     * between the center point and its furthest point in the neighborhood.
     */
    std::vector<int> radBound;
    /**
     * @brief Vector of integers specifying how many 2D sectors must be
     *  computed by the \f$i\f$-th filter. Note that zero means no 2D sectors
     *  will be computed. The 2D sectors will be linearly spaced. Each 2D
     *  sector filter will return the number of 2D sectors with at least one
     *  point (i.e., the count of non-empty 2D sectors).
     */
    std::vector<int> sect2D;
    /**
     * @brief Vector of integers specifying how many 3D sectors must be
     *  computed by the \f$i\f$-th filter. Note that zero means no 3D sectors
     *  will be computed. The 3D sectors will be linearly spaced. Each 3D
     *  sector filter will return the number of 3D sectors with at least one
     *  point (i.e., the count of non-empty 3D sectors).
     */
    std::vector<int> sect3D;
    /**
     * @brief The indices of the features involved in each filter.
     *
     * For example, condFeatIndices[i][j] gives the index of the \f$j\f$-th
     * feature involved in the computation of the \f$i\f$-th filter.
     */
    std::vector<std::vector<arma::uword>> condFeatIndices;
    /**
     * @brief The types of the conditions involved in each filter.
     *
     * For example, condTypes[i][j] gives the index of the \f$j\f$-th condition
     * involved in the computation of the \f$i\f$-th filter.
     */
    std::vector<std::vector<std::string>> condTypes;
    /**
     * @brief The target values of the conditions involved in each filter.
     *
     * For example, condTargets[i][j][k] gives the index of the \f$k\f$-th
     * target value of the \f$j\f$-th condition
     * involved in the computation of the \f$i\f$-th filter.
     */
    std::vector<std::vector<std::vector<FDecimalType>>> condTargets;
    /**
     * @brief The number of features (\f$n_f\f$) that must be computed.
     *
     * Note that the number of features does not necessarily match the number
     * of input features because the number of recount-based features that must
     * be computed can be different than the number of input features.
     */
    arma::uword numFeatures;
    /**
     * @brief How many threads must be used.
     *
     * If 0, then the number of threads depends on the current configuration.
     * If -1, then all available threads will be used.
     */
    int nthreads;

    // ***   ATTRIBUTES : STATE   *** //
    // ****************************** //
    /**
     * @brief The function to extract the neighborhood.
     */
    std::function<arma::Col<arma::uword>(
        arma::subview_row<XDecimalType> const &xi,
        arma::subview_col<XDecimalType> const &z,
        KDTree<arma::uword, XDecimalType> &kdt,
        arma::uword const k,
        arma::Col<XDecimalType> const &radius,
        XDecimalType const lowerBound,
        XDecimalType const upperBound
    )> query;
    /**
     * @brief The functions to handle NaN values for each recount filter.
     */
    std::vector<std::function<void(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN
    )>> nanHandlingFunctions;
    /**
     * @brief The functions that compute the condition-based filtering before
     *  the computing the recount-based features.
     * @see RecountMiner::notEqualsCondFun
     * @see RecountMiner::equalsCondFun
     * @see RecountMiner::lessThanCondFun
     * @see RecountMiner::lessThanOrEqualToCondFun
     * @see RecountMiner::greaterThanCondFun
     * @see RecountMiner::greaterThanOrEqualToCondFun
     * @see RecountMiner::inCondFun
     * @see RecountMiner::notInCondFun
     */
    std::vector<std::vector<std::function<void(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    )>>> conditionFunctions;
    /**
     * @brief The functions that compute the recount-based features (i.e.,
     *  the recounts).
     * @see RecountMiner::recountAbsoluteFrequency
     * @see RecountMiner::recountRelativeFrequency
     * @see RecountMiner::recountSurfaceDensity
     * @see RecountMiner::recountVolumeDensity
     * @see RecountMiner::recountVerticalSegments
     * @see RecountMiner::recountRings
     * @see RecountMiner::recountRadialBoundaries
     * @see RecountMiner::recountSectors2D
     * @see RecountMiner::recountSectors3D
     */
    std::vector<std::vector<std::function<FDecimalType(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    )>>> recountFunctions;
    /**
     * @brief The radius (\f$r\f$) parameter governing each recount.
     */
    std::vector<std::vector<XDecimalType>> recountRadius;
    /**
     * @brief The integer (\f$K\f$) parameter governing each recount.
     */
    std::vector<std::vector<int>> recountK;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build a recount miner.
     * @see nbhType
     * @see nbhK
     * @see nbhRadius
     * @see nbhLowerBound
     * @see nbhUpperBound
     * @see ignoreNan
     * @see absFreq
     * @see relFreq
     * @see surfDens
     * @see volDens
     * @see vertSeg
     * @see rings
     * @see radBound
     * @see sect2D
     * @see sect3D
     * @see condFeatIndices
     * @see condTypes
     * @see condTargets
     * @see nthreads
     */
    RecountMiner(
        string const &nbhType,
        int const nbhK,
        arma::Col<XDecimalType> const nbhRadius,
        XDecimalType const nbhLowerBound,
        XDecimalType const nbhUpperBound,
        std::vector<bool> const &ignoreNan,
        std::vector<bool> const &absFreq,
        std::vector<bool> const &relFreq,
        std::vector<bool> const &surfDens,
        std::vector<bool> const &volDens,
        std::vector<int> const &vertSeg,
        std::vector<int> const &rings,
        std::vector<int> const &radBound,
        std::vector<int> const &sect2D,
        std::vector<int> const &sect3D,
        std::vector<std::vector<arma::uword>> const &condFeatIndices,
        std::vector<std::vector<std::string>> const &condTypes,
        std::vector<std::vector<std::vector<FDecimalType>>> const &condTargets,
        int const nthreads
    );
    virtual ~RecountMiner() = default;

    // ***  DATA MINING METHODS  *** //
    // ***************************** //
    /**
     * @brief Mine the recount-based features from the input point cloud.
     * @param X The structure space representing the input point cloud whose
     *  recount-based features must be computed.
     * @param F The feature space representing the input point cloud whose
     *  recount-based features must be computed.
     * @return The \f$n_f\f$ mined counts
     *  \f$\pmb{\hat{F}} \in \mathbb{R}^{m \times n_f}\f$.
     */
    arma::Mat<FDecimalType> mine(
        arma::Mat<XDecimalType> const &X,
        arma::Mat<FDecimalType> const &F
    );
    /**
     * @brief Compute the recount-based features for the \f$i\f$-th point.
     *
     * @param[in] xi The point for which the recount-based features must be
     *  computed.
     * @param[in] i The index of the point whose recount-based features must be
     *  computed.
     * @param[in] X The structure space matrix representing the point cloud.
     * @param[in] F The feature space matrix representing the point cloud.
     * @param[in] kdt The KDTree used to compute the corresponding
     *  neighborhood for the given point.
     * @param[out] fout The output feature vector. The \f$i\f$-th row of this
     *  matrix will be written with the corresponding recount-based features.
     */
    void computeRecounts(
        arma::subview_row<XDecimalType> const &xi,
        arma::uword const i,
        arma::Mat<XDecimalType> const &X,
        arma::Mat<FDecimalType> const &F,
        KDTree<arma::uword, XDecimalType> &kdt,
        arma::subview_row<FDecimalType> fout
    ) const;

    // ***   CONDITION METHODS   *** //
    // ***************************** //
    /**
     * @brief Preserve only those points satisfying the condition:
     *
     * \f[
     *  f_{ij} \neq y
     * \f]
     *
     * Where \f$f_{ij}\f$ is the \f$j\f$-th feature of the \f$i\f$-th point
     * and \f$y\f$ is the target value.
     *
     * @param XN The structure space matrix representing the neighborhood.
     * @param FN The feature space matrix representing the neighborhood.
     * @param fIdx The feature index (\f$j\f$).
     * @param target The target value (\f$y\f$).
     */
    static void notEqualsCondFun(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    );
    /**
     * @brief Preserve only those points satisfying the condition:
     *
     * \f[
     *  f_{ij} = y
     * \f]
     *
     * Where \f$f_{ij}\f$ is the \f$j\f$-th feature of the \f$i\f$-th point
     * and \f$y\f$ is the target value.
     *
     * @see RecountMiner::notEqualsCondFun
     */
    static void equalsCondFun(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    );
    /**
     * @brief Preserve only those points satisfying the condition:
     *
     * \f[
     *  f_{ij} < y
     * \f]
     *
     * Where \f$f_{ij}\f$ is the \f$j\f$-th feature of the \f$i\f$-th point
     * and \f$y\f$ is the target value.
     *
     * @see RecountMiner::notEqualsCondFun
     */
    static void lessThanCondFun(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    );
    /**
     * @brief Preserve only those points satisfying the condition:
     *
     * \f[
     *  f_{ij} \leq y
     * \f]
     *
     * Where \f$f_{ij}\f$ is the \f$j\f$-th feature of the \f$i\f$-th point
     * and \f$y\f$ is the target value.
     *
     * @see RecountMiner::notEqualsCondFun
     */
    static void lessThanOrEqualToCondFun(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    );
    /**
     * @brief Preserve only those points satisfying the condition:
     *
     * \f[
     *  f_{ij} > y
     * \f]
     *
     * Where \f$f_{ij}\f$ is the \f$j\f$-th feature of the \f$i\f$-th point
     * and \f$y\f$ is the target value.
     *
     * @see RecountMiner::notEqualsCondFun
     */
    static void greaterThanCondFun(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    );
    /**
     * @brief Preserve only those points satisfying the condition:
     *
     * \f[
     *  f_{ij} \geq y
     * \f]
     *
     * Where \f$f_{ij}\f$ is the \f$j\f$-th feature of the \f$i\f$-th point
     * and \f$y\f$ is the target value.
     *
     * @see RecountMiner::notEqualsCondFun
     */
    static void greaterThanOrEqualToCondFun(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    );
    /**
     * @brief Preserve only those points satisfying the condition:
     *
     * \f[
     *  f_{ij} \in y
     * \f]
     *
     * Where \f$f_{ij}\f$ is the \f$j\f$-th feature of the \f$i\f$-th point
     * and \f$y\f$ is the target value.
     *
     * @see RecountMiner::notEqualsCondFun
     */
    static void inCondFun(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    );
    /**
     * @brief Preserve only those points satisfying the condition:
     *
     * \f[
     *  f_{ij} \notin y
     * \f]
     *
     * Where \f$f_{ij}\f$ is the \f$j\f$-th feature of the \f$i\f$-th point
     * and \f$y\f$ is the target value.
     *
     * @see RecountMiner::notEqualsCondFun
     */
    static void notInCondFun(
        arma::Mat<XDecimalType> &XN,
        arma::Mat<FDecimalType> &FN,
        arma::uword const fIdx,
        std::vector<FDecimalType> const &target
    );


    // ***   RECOUNT METHODS   *** //
    // *************************** //
    /**
     * @brief Count the number of points inside the neighborhood
     * @param xi The center point of the neighborhood.
     * @param XN The structure space of the neighborhood, i.e., the matrix
     *  of point-wise coordinates representing the neighborhood.
     * @param FN The feature space of the neighborhood, i.e.,, the matrix
     *  of features representing the neighborhood.
     * @param r The radius defining the neighborhood, if any.
     * @param K An integer number governing the recount operation, e.g., the
     *  total number of points.
     * @return The result of the recount. Whether an integer (e.g., cardinality
     *  counts) or a decimal (e.g., density) it is returned with the same
     *  data type as the feature space.
     */
    static FDecimalType recountAbsoluteFrequency(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );
    /**
     * @brief The relative frequency, i.e., the absolute frequency divided
     *  by the total number of points before applying the conditions.
     * @param K The total number of points in the neighborhood before applying
     *  the conditions.
     * @return The relative frequency.
     * @see RecountMiner::recountAbsoluteFrequency
     */
    static FDecimalType recountRelativeFrequency(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );
    /**
     * @brief The surface density, i.e., the number of points divided by the
     *  surface area.
     * @param r The radius of the neighborhood. When the neighborhood is not
     *  defined by a radius (e.g., KNN), then \f$r\f$ should be zero so the
     *  radius of the neighborhood is determined as the distance between the
     *  center point and the furthest point in the neighborhood.
     * @return The surface density in \f$\text{pts}/m^2\f$.
     */
    static FDecimalType recountSurfaceDensity(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );
    /**
     * @brief The volume density, i.e., the number of points divided by the
     *  spatial volume.
     * @param r The radius of the neighborhood. When the neighborhood is not
     *  defined by a radius (e.g., KNN), then \f$r\f$ should be zero so the
     *  radius of the neighborhood is determined as the distance between the
     *  center point and the furthest point in the neighborhood.
     * @param K It should be zero in general but two for 2D neighborhoods so
     *  volume is properly computed by propagating the corresponding 2D area
     *  through the \f$z\f$-axis. Typically the area of the circle is computed
     *  and then the volume is obtained assuming it is propagated along the
     *  segment starting at the min \f$z\f$ coordinate and ending at the max
     *  \f$z\f$ coordinate.
     * @return The volume density in \f$\text{pts}/m^3\f$.
     */
    static FDecimalType recountVolumeDensity(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );
    /**
     * @brief Divide the \f$z\f$ axis in linearly spaced intervals and count
     *  how many of them are populated (i.e., have at least one point)
     *  considering the points in the neighborhood.
     * @param K How many vertical segments must be computed.
     * @return The number of non-empty vertical segments.
     */
    static FDecimalType recountVerticalSegments(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );
    /**
     * @brief Compute concentric ring-like boundaries centered in the
     *  neighborhood's center point and count how many of them are populated
     *  (i.e., have at least one point of the neighborhood inside).
     *
     * NOTE that here ring is used as a synonym of annulus and not with the
     * meaning of an algebraic ring (scalar field without multiplicative
     * inverses).
     *
     * @param K How many rings must be computed.
     * @return The number of non-empty rings (annulus).
     * @see RecountMiner::recountRadialBoundaries
     */
    static FDecimalType recountRings(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );
    /**
     * @brief Compute the 3D version of the RecountMiner::recountRings method
     *  that works with spherical shells instead of annuli.
     * @param K How many spherical shells must be computed.
     * @return The number of non-empty spherical shells.
     * @see RecountMiner::recountRings
     */
    static FDecimalType recountRadialBoundaries(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );
    /**
     * @brief Divide the horizontal plane into sectors and count how many of
     *  them are non-empty.
     * @param K The number of 2D sectors to be computed.
     * @return The number of non-empty 2D sectors on the horizontal plane
     *  (the one considering \f$x\f$ and \f$y\f$ coordinates).
     */
    static FDecimalType recountSectors2D(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );
    /**
     * @brief Divide the space into 3D sectors and count how many of them
     *  are non-empty.
     * @param K The number of 3D sectors to be computed. It will be divided
     *  into \f$K_1\f$ polar partitions (\f$z\f$ coordinate) and \f$K_2\f$
     *  azimuth partitions (\f$x\f$ and \f$y\f$ coordinates), with
     *  \f$K_1 = \left\lceil\sqrt{K}\right\rceil\f$ and
     *  \f$K_2 = \left\lceil\frac{K}{K_1}\right\rceil\f$.
     * @return The number of non-empty 3D sectors.
     */
    static FDecimalType recountSectors3D(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        XDecimalType r,
        int const K
    );


};

#include <mining/RecountMiner.tpp>

}


#endif