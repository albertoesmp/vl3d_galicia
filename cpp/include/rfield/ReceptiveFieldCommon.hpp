#ifndef VL3DPP_RECEPTIVE_FIELD_COMMON_
#define VL3DPP_RECEPTIVE_FIELD_COMMON_

// ***   INCLUDES   *** //
// ******************** //
#include <armadillo>

namespace vl3dpp::rfield{


// ***  PROPAGATION METHODS  *** //
// ***************************** //
/**
 * @brief Propagate \f$\pmb{F} \in \mathbb{R}^{R \times n_f}\f$ values from the
 *  receptive field to \f$\pmb{Y} \in \mathbb{R}^{m \times n_f}\f$ values
 *  in the original space considering the \f$K\f$-nearest neigbhors for each
 *  of the \f$m \in \mathbb{Z}_{>0}\f$ points in the original space. The
 *  neighborhoods are codified through a matrix
 *  \f$\pmb{M} \in \mathbb{Z}^{m \times K}\f$ such that \f$m_{ik}\f$ is the
 *  \f$k\f$-th neighbor in the receptive field of the \f$i\f$-th point in the
 *  original space. The values from the receptive field are propagated
 *  considering their mean, i.e.:
 *
 *  \f[
 *      y_{ij} = K^{-1} \sum_{k=1}^{K} f_{m_{ik}j}
 *  \f]
 *
 * @tparam FDecimalType The decimal type for the feature space.
 * @tparam IndexType The type for neighborhood indexing.
 * @param M The decoding neighborhood matrix.
 * @param F The feature space matrix.
 * @return The propagated feature space.
*/
template <typename FDecimalType, typename IndexType>
arma::Mat<FDecimalType> propagateMean(
    arma::Mat<IndexType> const &M,
    arma::Mat<FDecimalType> const &F
);

/**
 * @brief Propagate \f$\pmb{F} \in \mathbb{R}^{R \times n_f}\f$ values from the
 *  receptive field to \f$\pmb{Y} \in \mathbb{R}^{m \times n_f}\f$ values
 *  in the original space considering the \f$K\f$-nearest neigbhors for each
 *  of the \f$m \in \mathbb{Z}_{>0}\f$ points in the original space. The
 *  neighborhoods are codified through a matrix
 *  \f$\pmb{M} \in \mathbb{Z}^{m \times K}\f$ such that \f$m_{ik}\f$ is the
 *  \f$k\f$-th neighbor in the receptive field of the \f$i\f$-th point in the
 *  original space. The values from the receptive field are propagated
 *  considering their mean, i.e.:
 *
 *  \f[
 *      y_{ij} = f_{m_{i1}j}
 *  \f]
 *
 * @tparam FDecimalType The decimal type for the feature space.
 * @tparam IndexType The type for neighborhood indexing.
 * @param M The decoding neighborhood matrix.
 * @param F The feature space matrix.
 * @return The propagated feature space.
*/
template <typename FDecimalType, typename IndexType>
arma::Mat<FDecimalType> propagateClosest(
    arma::Mat<IndexType> const &M,
    arma::Mat<FDecimalType> const &F
);


// ***  REDUCTION METHODS  *** //
// *************************** //
/**
 * @brief Reduce \f$\pmb{F} \in \mathbb{R}^{m \times n_f}\f$ values from the
 *  original space to \f$\pmb{Y} \in \mathbb{R \times n_f}\f$ values in the
 *  receptive field space considering the \f$K\f$-nearest neighbors for each of
 *  the \f$R \in \mathbb{Z}_{>0}\f$ points in the receptive field. The
 *  neighborhoods are codified through a matrix
 *  \f$\pmb{N} \in \mathbb{Z}^{R \times K}\f$ such that \f$n_{ik}\f$ is the
 *  \f$k\f$-th neighbor in the original space of the \f$i\f$-th point in the
 *  receptive field. The values from the receptive field are reduced
 *  considering their mean, i.e.:
 *
 *  \f[
 *      y_{ij} = K^{-1} \sum_{k=1}^{K} f_{m_{ik}j}
 *  \f]
 *
 * @tparam FDecimalType The decimal type for the feature space.
 * @tparam IndexType The type for neighborhood indexing.
 * @param N The encoding neighborhood matrix.
 * @param F The feature space matrix.
 * @return The reduced feature space.
 */
template <typename FDecimalType, typename IndexType>
arma::Mat<FDecimalType> reduceMean(
    arma::Mat<IndexType> const &N,
    arma::Mat<FDecimalType> const &F
);

/**
 * @brief Reduce \f$\pmb{y} \in \mathbb{Z}^{m}\f$ labels from the original
 *  space to \f$\pmb{y'} \in \mathbb{R}^{R}\f$ labels in the receptive field
 *  space considering the \f$K\f$-nearest neighbors for each of the
 *  \f$R \in \mathbb{Z}_{>0}\f$ points in the receptive field. The neighborhoods
 *  are codified through a matrix \f$\pmb{N} \in \mathbb{Z}_{>0}\f$ points in
 *  the receptive field. The neighborhoods are codified through a matrix
 *  \f$\pmb{N} \in \mathbb{Z}^{R \times K}\f$ such that \f$n_{ik}\f$ is the
 *  \f$k\f$-th neighbor in the original space of the \f$i\f$-th point in the
 *  receptive field. The values from the receptive field are reduced
 *  considering the mode, i.e., the most frequent class is selected.
 * @tparam LabelType The type for the class labels.
 * @tparam IndexType The type for neighborhood indexing.
 */
template <typename LabelType, typename IndexType>
arma::Col<LabelType> reduceLabelMode(
    arma::Mat<IndexType> const &N,
    arma::Col<LabelType> const &y,
    LabelType const ny
);

/**
 * @brief Like the vl3dpp::rfield::reduceLabelMode method but considering the
 *  label of the closest neighbor instead of the neighborhood's mode.
 * @see vl3dpp::rfield::reduceLabelMode
 */
template <typename LabelType, typename IndexType>
arma::Col<LabelType> reduceLabelClosest(
    arma::Col<IndexType> const &N,
    arma::Col<LabelType> const &y,
    LabelType const ny
);


// ***  DECODING REDUCTION METHODS  *** //
// ************************************ //
/**
 * @brief Reduce the encoded features (e.g., probabilities)
 *  \f$\mathcal{E} = \left\{\pmb{E_t} \in \mathbb{R}^{R_t \times n_f} \right\}_{t=1}^{\text{bs}}\f$
 *  for a batch of \f$\text{bs}\f$ matrices, each representing a receptive
 *  field with \f$R_t\f$ rows and \f$n_c\f$ columns, to the decoded point-wise
 *  features matrix \f$\pmb{Z} \in \mathbb{R}^{m \times n_f}\f$ representing
 *  the values of \f$n_f\f$ point-wise features on the original
 *  \f$m \in \mathbb{Z}_{>0}\f$ input points. The resulting values are given
 *  by the following expression:
 *
 * \f[
 *  z_{ij} = \left(
 *      \sum_{t=1}^{\text{bs}}{\left|{\mathcal{N}(t, i)}\right|}
 *  \right)^{-1} \sum_{t=1}^{\text{bs}}{
 *      \sum_{k_i \in \mathcal{N}(t, i)}(\pmb{E_t})_{k_ij}
 *  }
 * \f]
 *
 * Where \f$z_{ij}\f$ is the \f$j\f$-th decoded feature for the \f$i\f$-th
 *  original point, \f$\mathcal{N}_(t, i)\f$ is the set of indices representing
 *  the neighbors of the \f$i\f$-th original point in the \f$t\f$-th receptive
 *  field, and \f$(\pmb{E_t})_{k_ij}\f$ is the \f$j\f$-th feature of the
 *  \f$k_i\f$ neighbor in the \f$t\f$-th receptive field of the \f$i\f$-th
 *  original point
 *
 * @tparam FDecimalType The decimal type for the feature space.
 * @tparam IndexType The index type for neighborhood indexing.
 * @param[in] I The \f$\text{bs}\f$ decoding neighborhoods of \f$m_t\f$ points each.
 * @param[in] outColIdx The column where the output must be written (so it can be
 *  parallelized for each feature).
 * @param[in] encoded The encoded features for each receptive field.
 * @param[out] out Where to write the output.
 */
template <typename FDecimalType, typename IndexType>
void decodingReduceMean(
    std::vector<arma::Col<IndexType>> const &I,
    arma::uword const outColIdx,
    std::vector<arma::Mat<FDecimalType>> const &encoded,
    arma::Mat<FDecimalType> &out
);


/**
 * @brief Like rfield::decodingReduceMean but the output is computed such that:
 *
 * \f[
 *  z_{ij} = \sum_{t=1}^{\text{bs}}{
 *      \sum_{k_i \in \mathcal{N}(t, i)}(\pmb{E_t})_{k_ij}
 *  }
 * \f]
 *
 * @see vl3dpp::rfield::decodingReduceMean
 */
template <typename FDecimalType, typename IndexType>
void decodingReduceSum(
    std::vector<arma::Col<IndexType>> const &I,
    arma::uword const outColIdx,
    std::vector<arma::Mat<FDecimalType>> const &encoded,
    arma::Mat<FDecimalType> &out
);

/**
 * @brief Like rfield::decodingReduceMean but the output is computed such that:
 *
 * \f[
 *  z_{ij} = \max_{t, i} \quad \biggl\{{
 *      \Bigl(\pmb{E_t}\Bigr)_{k_ij} :
 *      k_i \in \mathcal{N}(t, i)
 *  }\biggr\}
 * \f]
 *
 * @see vl3dpp::rfield::decodingReduceMean
 */
template <typename FDecimalType, typename IndexType>
void decodingReduceMax(
    std::vector<arma::Col<IndexType>> const &I,
    arma::uword const outColIdx,
    std::vector<arma::Mat<FDecimalType>> const &encoded,
    arma::Mat<FDecimalType> &out
);

/**
 * @brief Like rfield::decodingReduceMean but the output is computed as
 *  explained in the Python documentation for
 *  src.utils.preds.entropic_pred_reduce_strategy.EntropicPredReduceStrategy
 *
 * @param[in] W The point-wise weights.
 * @param[out] lazyNorm Where the point-wise norms (denominators) will be
 *  stored.
 * @see vl3dpp::rfield::decodingReduceMean
 */
template <typename FDecimalType, typename IndexType>
void decodingReduceLazyWeightedMean(
    std::vector<arma::Col<IndexType>> const &I,
    arma::uword const outColIdx,
    std::vector<arma::Mat<FDecimalType>> const &encoded,
    std::vector<arma::Col<FDecimalType>> const &W,
    arma::Mat<FDecimalType> &out,
    arma::Col<FDecimalType> &lazyNorm
);


// ***  TRANSFORMATIONS  *** //
// ************************* //
/**
 * @brief Transform the given structure space scaling it to the unit sphere.
 *
 * The center point of the unit sphere is assumed to be at zero, i.e., X must
 * be centered at zero. The radius of the sphere is considered as the distance
 * between zero and the furthest point in X.
 *
 * @param X The structure space to be scaled to the unit sphere.
 * @return Nothing, the input receptive field is updated in place.
 */
template <typename XDecimalType>
void toUnitSphere(arma::Mat<XDecimalType> &X);


#include <rfield/ReceptiveFieldCommon.cpp>


}
#endif