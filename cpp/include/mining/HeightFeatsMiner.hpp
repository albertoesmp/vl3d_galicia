#ifndef VL3DPP_HEIGHT_FEATS_MINER_
#define VL3DPP_HEIGHT_FEATS_MINER_

// ***   INCLUDES   *** //
// ******************** //
#include <adt/kdtree/KDTree.hpp>
#include <adt/grid/LazySupportGrid.hpp>
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

namespace vl3dpp::mining {

using vl3dpp::adt::kdtree::KDTree;
using vl3dpp::adt::grid::LazySupportGrid;

using std::vector;
using std::string;


// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing a data miner for height features.
 *
 * @tparam XDecimalType The data type for the decimal numbers in the context
 *  of the point cloud's structure space.
 * @tparam FDecimalType The data type for the decimal numbers in the context
 *  of the point cloud's feature space.
 */
template <typename XDecimalType, typename FDecimalType>
class HeightFeatsMiner {
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief String specifying the type of neighborhood. It can be either
     *  "cylinder" or "rectangular2d".
     */
    string nbhType;
    /**
     * @brief Either the radius of the cylinder disk or half the side of a
     *  rectangular region.
     */
    XDecimalType nbhRadius;
    /**
     * @brief How many times the radius for the separation between support
     *  points.
     *
     * When the separation factor is strictly greater than zero,
     * the height features will be computed for the corresponding support
     * points and then propagated to the input point cloud.
     *
     * Note that the ceil and floor distance will be computed point-wise
     * considering the max and min vertical coordinates from the corresponding
     * support point.
     *
     * When the separation factor is strictly zero, the height features will be
     * directly computed point-wise.
     *
     * @see grid::LazySupportGrid
     */
    XDecimalType nbhSeparationFactor;
    /**
     * @brief The outlier filter to be applied, if any. It can be "iqr" for
     *  an outlier filter based on the quartiles or "stdev" for an outlier
     *  filter based on the standard deviation. If the empty string "" is
     *  given, then no outlier filter will be computed at all.
     *
     * The IQR filter for quartiles \f$Q_1 < Q_2 < Q_3\f$ filters out any value
     * that is not inside the following interval:
     *
     * \f[
     *  \left[
     *      Q_1 - \frac{3}{2} (Q_3-Q_1) ,\,
     *      Q_3 + \frac{3}{2} (Q_3-Q_1)
     *  \right]
     * \f]
     *
     * The standard deviation filter for mean \f$\mu \in \mathbb{R}\f$ and
     * standard deviation \f$\sigma \in \mathbb{R}\f$ filters out any value
     * that is not inside the following interval:
     *
     * \f[
     *  \left[\mu - 3 \sigma ,\, \mu + 3 \sigma\right]
     * \f]
     */
    string outlierFilter;
    /**
     * @brief Vector with the name of the height features to be computed.
     *
     * Supported height features are:
     *
     * <ul>
     *  <li>"floor_distance"</li>
     *  <li>"ceil_distance"</li>
     *  <li>"floor_coordinate"</li>
     *  <li>"ceil_coordinate"</li>
     *  <li>"height_range"</li>
     *  <li>"mean_height"</li>
     *  <li>"median_height"</li>
     *  <li>"height_quartiles"</li>
     *  <li>"height_deciles"</li>
     *  <li>"height_variance"</li>
     *  <li>"height_stdev"</li>
     *  <li>"height_skewness"</li>
     *  <li>"height_kurtosis"</li>
     * </ul>
     */
    vector<string> fnames;
    /**
     * @brief The number of features (\f$n_f\f$) that must be computed.
     *
     * Note that the number of features does not necessarily match the number
     * of feature names. They will match if only scalar features are requested.
     * However, if quartiles are requested, a 3-dimensional vector will be
     * generated instead of a scalar.
     */
    arma::uword numFeatures;
    /**
     * @brief How many threads must be used.
     *
     * If 0, then the number of threads depends on the current configuration.
     * If -1, then all available threads will be used.
     */
    int nthreads;

    // ***  ATTRIBUTES : STATE  *** //
    // **************************** //
    /**
     * @brief The grid representing the support points.
     *
     * @see HeightFeatsMiner::nbhSeparationFactor
     */
    LazySupportGrid<XDecimalType> * supportGrid;
    /**
     * @brief The function to filter the outliers.
     */
    std::function<void(arma::Col<FDecimalType> &)> outlierFilterFunction;
    /**
     * @brief The function to find the requested neighborhoods.
     */
    std::function<arma::Col<arma::uword>(
        KDTree<arma::uword, XDecimalType> &,
        arma::Row<XDecimalType> const &,
        XDecimalType const r
    )> findNeighborhoodFunction;
    /**
     * @brief The type definition that represents any height feature function.
     * @see HeightFeatsMiner::heightFeaturesFunctions
     */
    typedef std::function<void(
        arma::Row<XDecimalType> const &,
        arma::Col<FDecimalType> const &,
        bool &,
        bool &,
        bool &,
        bool &,
        FDecimalType &,
        FDecimalType &,
        FDecimalType &,
        FDecimalType &,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    )> HeightFeatureFunction;
    /**
     * @brief The functions that compute the requested height features.
     * @see HeightFeatsMiner::heightFeatureFunction
     */
    std::vector<HeightFeatureFunction> heightFeaturesFunctions;


public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build a height features miner.
     * @see nbhType
     * @see nbhRadius
     * @see nbhSeparationFactor
     * @see outlierFilter
     * @see fnames
     * @see nthreads
     */
    HeightFeatsMiner(
        string const &nbhType,
        XDecimalType const nbhRadius,
        XDecimalType const nbhSeparationFactor,
        string const &outlierFilter,
        vector<string> const &fnames,
        int const nthreads=0
    );
    virtual ~HeightFeatsMiner() = default;


    // ***  DATA MINING METHODS  *** //
    // ***************************** //
    /**
     * @brief Mine the height features on the given structure space.
     *
     * @param X The structure space matrix
     *  \f$\pmb{X} \in \mathbb{R}^{m \times 3}\f$ whose \f$m\f$ rows represent
     *  the point-wise coordinates of a 3D point cloud.
     * @return The \f$n_f\f$ mined point-wise height features
     *  \f$\pmb{F} \in \mathbb{R}^{m \times n_f}\f$.
     */
    arma::Mat<FDecimalType> mine(arma::Mat<XDecimalType> const &X);

    /**
     * @brief Compute the height features for the \f$i\f$-th point.
     *
     * @param[in] x The point for which the height features must be computed.
     * @param[in] i The index of the point whose height features must be
     *  computed.
     * @param[in] X The structure space matrix representing the point cloud.
     * @param[in] kdt The KDTree to speedup the spatial queries.
     * @param[out] F The output feature matrix. The \f$i\f$-th row of this
     *  matrix will be written with the corresponding height features.
     */
    void computeHeightFeatures(
        arma::Row<XDecimalType> const &x,
        arma::uword const i,
        arma::Mat<XDecimalType> const &X,
        KDTree<arma::uword, XDecimalType> &kdt,
        arma::Mat<FDecimalType> &F
    ) const;


    // ***  OUTLIER FILTERING METHODS  *** //
    // *********************************** //
    /**
     * @brief Function providing the logic of the IQR filter for vertical
     *  values.
     *
     * @param z The vertical values to be filtered (inplace).
     * @return Filtered vertical values.
     * @see HeightFeatsMiner::outlierFilter
     */
    static void filterByIQR(arma::Col<FDecimalType> &z);
    /**
     * @brief Function providing the logic of the IQR filter for vertical
     *  values.
     *
     * @param z The vertical values to be filtered (inplace).
     * @see HeightFeatsMiner::outlierFilter
     */
    static void filterByStdev(arma::Col<FDecimalType> &z);

    // ***  HEIGHT FEATURES METHODS  *** //
    // ********************************* //
    /**
     * @brief Compute the coordinate of the lowest point in the neighborhood.
     *
     * @param xi The point whose floor coordinate must be computed.
     * @param z The vertical coordinates in the neighborhood of the point
     *  whose floor coordinate must be computed.
     * @param minComputed Boolean flag specifying whether the min vertical
     *  value has been computed for the current neighborhood (true) or not
     *  (false).
     * @param maxComputed Boolean flag specifying whether the max vertical
     *  value has been computed for the current neighborhood (true) or not
     *  (false).
     * @param meanComputed Boolean flag specifying whether the mean vertical
     *  value has been computed for the current neighborhood (true) or not
     *  (false).
     * @param stdevComputed Boolean flag specifying whether the standard
     *  deviation of the vertical coordinates has been computed for the current
     *  neighborhood (true) or not (false).
     * @param min The min vertical value for the current neighborhood.
     * @param max The max vertical value for the current neighborhood.
     * @param mean The mean vertical value for the current neighborhood.
     * @param stdev The standard vertical value for the current neighborhood.
     * @param nextFeatIdx The index of the next feature to be computed. It
     *  must be updated after calling this method.
     * @param out The output features vector corresponding to the point whose
     *  neighborhood is being analyzed.
     */
    static void computeFloorCoordinate(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the coordinate of the highest point in the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeCeilCoordinate(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the vertical distance of the point to the lowest point
     *  in its neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeFloorDistance(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the vertical distance of the point to the highest point
     *  in its neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeCeilDistance(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the height range, i.e., the distance between the highest
     *  and the lowest points in the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeHeightRange(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the mean height value of the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeMeanHeight(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the median height value of the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeMedianHeight(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the quartiles of the height values of the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeHeightQuartiles(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the deciles of the height values of the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeHeightDeciles(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the variance of the height values of the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeHeightVariance(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the standard deviation of the height values of the
     *  neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeHeightStdev(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the skewness of the height values of the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeHeightSkewness(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );
    /**
     * @brief Compute the kurtosis of the height values of the neighborhood.
     * @see HeightFeatsMiner::computeFloorCoordinate
     */
    static void computeHeightKurtosis(
        arma::Row<XDecimalType> const &xi,
        arma::Col<FDecimalType> const &z,
        bool &minComputed,
        bool &maxComputed,
        bool &meanComputed,
        bool &stdevComputed,
        FDecimalType &min,
        FDecimalType &max,
        FDecimalType &mean,
        FDecimalType &stdev,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> &out
    );

};

#include <mining/HeightFeatsMiner.tpp>

}

#endif