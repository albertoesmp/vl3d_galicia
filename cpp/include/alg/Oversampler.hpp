#ifndef VL3DPP_OVERSAMPLER_
#define VL3DPP_OVERSAMPLER_

// ***   INCLUDES   *** //
// ******************** //
#include <util/VL3DPPException.hpp>
#include <alg/FurthestPointSubsampler.hpp>
#include <adt/kdtree/KDTree.hpp>

#include <armadillo>

#include <sstream>
#include <functional>

using vl3dpp::alg::FurthestPointSubsampler;

namespace vl3dpp::alg{

// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @brief Class providing oversampling algorithms.
 *
 * Oversampling a structure space \f$\pmb{X} \in \mathbb{R}^{m \times n_x}\f$
 * means transforming it to another structure space
 * \f$\pmb{X'} \in \mathbb{R}^{m' \times n_x}\f$ with \f$m' > m\f$.
 *
 * @tparam XDecimalType The type for the decimal numbers representing the
 *  structure space (point-wise coordinates).
 * @tparam IndexType The type for the indices.
 */
template <typename XDecimalType, typename IndexType>
class Oversampler{
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief The minimum acceptable number of points \f$m_*\f$. Input point
     * clouds with \f$m < m_*\f$ points will raise an exception as they are
     * considered unreliable.
     */
    IndexType minPoints;
    /**
     * @brief The target number of points \f$m^*\f$, i.e., input structure
     * spaces must be oversampled to have \f$m^*\f$ points.
     */
    IndexType targetPoints;
    /**
     * @brief The oversampling strategy, it must be one of the following:
     *
     * <ul>
     *  <li>nearest</li>
     *  <li>knn</li>
     *  <li>gaussian_knn</li>
     *  <li>spherical</li>
     *  <li>spherical_radiation</li>
     * </ul>
     */
    std::string strategy;
    /**
     * @brief The number of \f$K\f$-nearest neighbors for the "knn" and
     *  "gaussian_knn" strategies.
     */
    IndexType K;
    /**
     * @brief The radius \f$r\f$ for the spherical and spherical radiation
     *  strategies.
     */
    XDecimalType radius;

    // ***  ATTRIBUTES : STATE  *** //
    // **************************** //
    /**
     * @brief The oversampling method that corresponds to the strategy of the
     *  vl3dpp::alg::Oversampler instance.
     */
    arma::Mat<XDecimalType> (
        Oversampler<XDecimalType, IndexType>::*oversample
    )(
        IndexType const K,
        arma::Mat<XDecimalType> const &X
    ) const;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Insantiate an oversampler.
     * @see Oversampler::minPoints
     * @see Oversampler::strategy
     * @see Oversampler::K
     * @see Oversampler::radius
     */
    Oversampler(
         IndexType const minPoints,
         IndexType const targetPoints,
         std::string const &strategy,
         IndexType const K,
         XDecimalType const radius
     );
     virtual ~Oversampler() = default;

    // ***   CALLABLE   *** //
    // ******************** //
    /**
     * @brief Oversample the given structure space.
     * @param X The structure space that must be oversampled.
     * @return The oversampled structure space containing the points in the
     *  input X but also the new points.
     */
    arma::Mat<XDecimalType> operator()(arma::Mat<XDecimalType> const &X) const;

    // ***  OVERSAMPLING METHODS  *** //
    // ****************************** //
    /**
     * @brief Oversample by computing the closest neighbor (distinct to
     *  itself) for each point and considering the mid-range from the
     *  \f$K\f$ closer pairs of neighbors.
     * @param kappa How many points must be sampled.
     * @param X The structure space to be oversampled.
     * @return The oversampled structure space.
     */
    arma::Mat<XDecimalType> nearestOversample(
        IndexType const kappa,
        arma::Mat<XDecimalType> const &X
    ) const;

    /**
     * @brief Oversample by computing the \f$K\f$-nearest neighbors for each
     *  point and considering their centroid as the new point. If more points
     *  than needed are generated, then those with the smallest distance with
     *  respect to neighborhood center will be prioritized.
     * @param kappa How many points must be sampled.
     * @param X The structure space to be oversampled.
     * @return The oversampled structure space.
     */
    arma::Mat<XDecimalType> knnOversample(
        IndexType const kappa,
        arma::Mat<XDecimalType> const &X
    ) const;
    /**
     * @brief Oversample like Oversampler::knnOversample but using a Gaussian
     *  RBF when computing the centroids so the closer to the center the
     *  greater the contribution.
     * @param kappa How many points must be sampled.
     * @param X The structure space to be oversampled.
     * @return The oversampled structure space.
     */
    arma::Mat<XDecimalType> gaussianKnnOversample(
        IndexType const kappa,
        arma::Mat<XDecimalType> const &X
    ) const;
    /**
     * @brief Oversample by computing the spherical neighborhood for each point
     *  with radius \f$r\f$. The centroid of each spherical neighborhood is an
     *  oversampled point. If more points than needed are generated, then
     *  up to \f$K\f$ points are selected considering the FIFO orderinrg of
     *  their indices.
     * @param kappa How many points must be sampled.
     * @param X The structure space to be oversampled.
     * @return The oversampled structure space.
     */
    arma::Mat<XDecimalType> sphericalOversample(
        IndexType const kappa,
        arma::Mat<XDecimalType> const &X
    ) const;
    /**
     * @brief Oversample like Oversampler::sphericalOversample but computing
     *  the centroid using a Gaussian RBF.
     * @param kappa How many points must be sampled.
     * @param X The structure space to be oversampled.
     * @return The oversampled structure space.
     */
    arma::Mat<XDecimalType> sphericalRadiationOversample(
        IndexType const kappa,
        arma::Mat<XDecimalType> const &X
    ) const;

    // ***  GETTERs and SETTERs  *** //
    // ***************************** //
    /**
     * @see Oversampler::minPoints
     */
    IndexType getMinPoints() const {return minPoints;}
};

#include <alg/Oversampler.tpp>

}

#endif