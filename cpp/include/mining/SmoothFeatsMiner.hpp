#ifndef VL3DPP_SMOOTH_FEATS_MINER_
#define VL3DPP_SMOOTH_FEATS_MINER_

// ***   INCLUDES   *** //
// ******************** //
#include <adt/kdtree/KDTree.hpp>
#include <adt/octree/Octree.hpp>
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


namespace vl3dpp { namespace mining {

using vl3dpp::adt::kdtree::KDTree;
using vl3dpp::adt::octree::Octree;

using std::function;
using std::vector;
using std::string;

// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing a data miner for smooth features.
 *
 * @tparam XDecimalType The data type for the decimal numbers in the context
 *  of the point cloud's structure space.
 * @tparam FDecimalType The data type for the decimal numbers in the context
 *  of the point cloud's feature space.
 */
template <typename XDecimalType, typename FDecimalType>
class SmoothFeatsMiner {
protected:
    // ***  NEIGHBORHOOD ENGINE  *** //
    // ***************************** //
    /**
     * @brief The class handling the neighborhood extraction for the smooth
     *  features miner.
     */
    class NeighborhoodEngine {
    public:
        /**
         * @brief The union type for the advanced data structure used to
         *  speedup the spatial queries to extract the corresponding
         *  neighborhoods.
         */
        union Queryer{
            KDTree<arma::uword, XDecimalType> * kdt = nullptr;
            Octree<arma::uword, XDecimalType> * octree;
            Queryer() = default; // Default constructor
            Queryer(Queryer &&rhs){ // Move constructor
                if(rhs.kdt!=nullptr){
                    kdt = rhs.kdt;
                    rhs.kdt = nullptr;
                }
                if(rhs.octree!=nullptr){
                    octree = rhs.octree;
                    rhs.octree = nullptr;
                }
            }
            ~Queryer() {
                if(kdt!=nullptr){
                    delete kdt;
                    kdt = nullptr;
                }
                if(octree != nullptr){
                    delete octree;
                    octree = nullptr;
                }
            }
        };
        /**
         * @brief The advanced data structure used to speedup the spatial
         *  queries to extract the corresponding neighborhoods.
         */
        Queryer queryer;
        /**
         * @brief The function to extract the neighborhood.
         */
        std::function<arma::Col<arma::uword>(
            arma::Col<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            Queryer const &queryer,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        )> query;
    };

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
     * @brief The \f$\omega\f$ parameter for the weighted mean strategy.
     */
    FDecimalType const weightedMeanOmega;
    /**
     * @brief The \f$\omega\f$ parameter for the Gaussian RBF strategy.
     */
    FDecimalType const gaussianRbfOmega;
    /**
     * @brief The policy specifying how to handle NaN values in the feature
     *  space. It can be "propagate" to propagate NaN values or "replace" to
     *  replace NaN values by the corresponding mean.
     */
    string const nanPolicy;
    /**
     * @brief Vector with the name of the smooth features to be computed.
     *
     * Supported smooth features are:
     *
     * <ul>
     *  <li>mean</li>
     *  <li>weighted_mean</li>
     *  <li>gaussian_rbf</li>
     * </ul>
     */
    vector<string> fnames;
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
     * @brief The type definition that represents any smooth feature function.
     * @see SmoothFeatsMiner::smoothFeaturesFunctions
     */
    typedef function<void(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        FDecimalType const omega,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> out
    )> SmoothFeaturesFunction;
    /**
     * @brief The functions that compute the requested smooth features.
     * @see SmoothFeatsMiner::SmoothFeaturesFunction
     * @see SmoothFeatsMiner::computeMeanSmooth
     * @see SmoothFeatsMiner::computeWeightedMeanSmooth
     * @see SmoothFeatsMiner::computeGaussianRbfSmooth
     */
    vector<SmoothFeaturesFunction> smoothFeaturesFunctions;
    /**
     * @brief The omega parameter for each smooth feature function in
     *  SmoothFeatsMiner::smoothFeaturesFunctions.
     * @see SmoothFeatsMiner::smoothFeaturesFunctions
     */
    vector<FDecimalType> smoothFeaturesOmegas;
    /**
     * @brief The type definition that represents any NaN handling function.
     * @see SmoothFeatsMiner::nanHandlingFunction
     */
    typedef function<void(arma::Mat<FDecimalType> &FN)> NanHandlingFunction;
    /**
     * @brief The function that handles NaN values during the computation of
     *  smooth features.
     * @see SmoothFeaturesMiner::nanPropagate
     * @see SmoothFeaturesMiner::nanReplace
     */
    NanHandlingFunction nanHandlingFunction;
    /**
     * @brief The squared \f$\omega^2\f$ parameter for the Gaussian RBF
     * strategy.
     *
     * The squared version of the omega parameter is computed to avoid
     * redundant computations when using the Gaussian RBF smoothing strategy.
     */
    FDecimalType const squaredGaussianRbfOmega;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build a smooth features miner.
     * @see nbhType
     * @see nbhK
     * @see nbhRadius
     * @see nbhLowerBound
     * @see nbhUpperBound
     * @see weightedMeanOmega
     * @see gaussianRbfOmega
     * @see nanPolicy
     * @see fnames
     * @see nthreads
     */
    SmoothFeatsMiner(
        string const &nbhType,
        int const nbhK,
        arma::Col<XDecimalType> const nbhRadius,
        XDecimalType const nbhLowerBound,
        XDecimalType const nbhUpperBound,
        FDecimalType const weightedMeanOmega,
        FDecimalType const gaussianRbfOmega,
        string const &nanPolicy,
        vector<std::string> &fnames,
        int const nthreads
    );
    virtual ~SmoothFeatsMiner() = default;

    // ***  DATA MINING METHODS  *** //
    // ***************************** //
    /**
     * @brief Mine the smooth version of the given feature space.
     * @param X The structure space representing the input point cloud whose
     *  smooth features must be computed.
     * @param F The feature space representing the input point cloud whose
     *  features must be smoothed.
     * @return The \f$n_f\f$ mined point-wise smooth features
     *  \f$\pmb{\hat{F}} \in \mathbb{R}^{m \times n_f}\f$.
     */
    arma::Mat<FDecimalType> mine(
        arma::Mat<XDecimalType> const &X,
        arma::Mat<FDecimalType> const &F
    );

    /**
     * @brief Compute the smooth features for the \f$i\f$-th point.
     *
     * @param[in] xi The point for which the smooth features must be computed.
     * @param[in] i The index of the point whose smooth features must be
     *  computed.
     * @param[in] X The structure space matrix representing the point cloud.
     * @param[in] F The feature space matrix representing the point cloud.
     * @param[in] nbhEngine The neighborhood engine used to compute the
     *  corresponding neighborhood for the given point.
     * @param[out] fout The output feature vector. The \f$i\f$-th row of this
     *  matrix will be written with the corresponding smooth features.
     */
    void computeSmoothFeatures(
        arma::subview_row<XDecimalType> const &xi,
        arma::uword const i,
        arma::Mat<XDecimalType> const &X,
        arma::subview_col<XDecimalType> const &z,
        arma::Mat<FDecimalType> const &F,
        NeighborhoodEngine const &nbhEngine,
        arma::subview_row<FDecimalType> fout
    ) const;


    // ***  NEIGHBORHOOD ENGINE METHODS  *** //
    // ************************************* //
    /**
     * @brief Build the neighborhood engine for the given structure space.
     * @param X Structure space matrix for which the spatial queries must be
     *  computed.
     * @return The built neighborhood engine.
     * @see SmoothFeatsMiner::NeighborhoodEngine
     */
    NeighborhoodEngine buildNeighborhoodEngine(
        arma::Mat<XDecimalType> const &X
    );

    // ***  NAN POLICY METHODS  *** //
    // **************************** //
    /**
     * @brief Propagate the NaN values, i.e., let them be.
     * @param FN The feature space of a neighborhood whose NaN values must be
     *  propagated.
     */
    static void nanPropagate(arma::Mat<FDecimalType> &FN);
    /**
     * @brief Replace the NaN values by the mean value of the feature in the
     *  given neighborhood.
     * @param FN The feature space of a neighborhood whose NaN values must be
     *  replaced by the mean.
     */
    static void nanReplace(arma::Mat<FDecimalType> &FN);


    // ***  SMOOTH FEATURES METHODS  *** //
    // ********************************* //
    /**
     * @brief Compute the requested mean-based smooth feature.
     *
     * @param xi The point whose smooth feature must be computed.
     * @param XN The structure space matrix representing the neighborhood.
     * @param FN The feature space matrix representing the neighborhood.
     * @param omega The parameter governing the smooth feature (it is not
     *  considered for the mean).
     * @param nextFeatIdx The index of the next feature to be computed. It
     *  must be updated after calling this method.
     * @param out The output features vector corresponding to the point whose
     *  neighborhood is being analyzed.
     * @see SmoothFeatsMiner::smoothFeaturesFunctions
     */
    static void computeMeanSmooth(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        FDecimalType const omega,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> fout
    );
    /**
     * @brief Compute the weighted mean-based smooth feature.
     * @see SmoothFeatsMiner::computeMeanSmooth
     * @see SmoothFeatsMiner::smoothFeaturesFunctions
     */
    static void computeWeightedMeanSmooth(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        FDecimalType const omega,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> fout
    );
    /**
     * @brief Compute the Gaussian radial basis function-based smooth feature.
     * @see SmoothFeatsMiner::computeMeanSmooth
     * @see SmoothFeatsMiner::smoothFeaturesFunctions
     */
    static void computeGaussianRbfSmooth(
        arma::subview_row<XDecimalType> const &xi,
        arma::Mat<XDecimalType> const &XN,
        arma::Mat<FDecimalType> const &FN,
        FDecimalType const squaredOmega,
        arma::uword &nextFeatIdx,
        arma::subview_row<FDecimalType> fout
    );

};

#include <mining/SmoothFeatsMiner.tpp>

}}

#endif