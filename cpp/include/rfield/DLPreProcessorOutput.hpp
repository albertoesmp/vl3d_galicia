#pragma once

// ***   INCLUDES   *** //
// ******************** //
#include <armadillo>

#include <vector>


namespace vl3dpp::rfield{


// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing the output of a deep learning pre-processor.
 *
 * @tparam InputXDecimalType The data type for the decimal numbers representing
 *  the input structure spaces (typically used to store the centers of each
 *  receptive field with the same decimal precision than the original input).
 * @tparam OutputXDecimalType The data type for the decimal numbers representing
 *  the structure spaces.
 * @tparam FDecimalType The data type for the decimal numbers representing the
 *  feature spaces.
 * @tparam LabelType The data type for the values representing the point-wise
 *  labels (e.g., point-wise classes).
 * @tparam IndexType The index type used to encode the neighborhoods.
 *
 * @see vl3dpp::rfield::DLFPSPreProcessor
 * @see vl3dpp::rfield::DLHierarchicalFPSPreProcessor
 */
template <
    typename InputXDecimalType,
    typename OutputXDecimalType,
    typename FDecimalType,
    typename LabelType,
    typename IndexType
>
class DLPreProcessorOutput{
public:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief The center point for each receptive field.
     *
     * \f[
     *  \pmb{X} \in \mathbb{R}^{\text{bs} \times n_x}
     * \f]
     *
     * Where \f$\text{bs}\f$ is the number of receptive fields and \f$n_x\f$
     * is the dimensionality of the structure space.
     */
    arma::Mat<InputXDecimalType> xout;
    /**
     * @brief The output batches of structure spaces.
     *
     *  \f[
     *   \forall 1 \leq d \leq D, \;
     *   \mathcal{X}_d \in \mathbb{R}^{\text{bs} \times R_d \times n_x}
     *  \f]
     *
     * Where \f$D\f$ is the max depth of the receptive field, \f$\text{bs}\f$
     * is the number of receptive fields, \f$R_d\f$ is the number of points
     * per receptive field at depth \f$d\f$, and \f$n_x\f$ is the
     * dimensionality of the structure space.
     */
    std::vector<std::vector<arma::Mat<OutputXDecimalType>>> Xout;
    /**
     * @brief The output batch of feature spaces.
     *
     * \f[
     *  \mathcal{F} \in \mathbb{R}^{\text{bs} \times R \times n_f}
     * \f]
     *
     * Where \f$\text{bs}\f$ is the number of receptive fields, \f$R\f$ is the
     * number of points per receptive field at the first depth, and \f$n_f\f$
     * is the dimensionality of the feature space.
     */
    std::vector<arma::Mat<FDecimalType>> Fout;
    /**
     * @brief The output matrix of labels.
     *
     * \f[
     *  \pmb{Y} \in \mathbb{R}^{\text{bs} \times R}
     * \f]
     *
     * Where \f$\text{bf}\f$ is the number of receptive fields, an \f$R\f$ is
     * the number of points per receptive field at the first depth.
     */
    arma::Mat<LabelType> yout;
    /**
     * @brief The matrices of indices connecting the original point cloud with
     *  the receptive fields.
     *
     * \f[
     *  \forall 1 < i < \text{bs},\;
     *  \pmb{I_i} \in \mathbb{Z}^{m_i}
     * \f]
     *
     * Where \f$m_i \in \mathbb{Z}_{>0}\f$ is the number of points in the
     * original point cloud that belong to the neighborhood represented by the
     * \f$i\f$-th receptive field and \f$(\pmb{I_i})_{p}\f$ can be read as the
     * index representing the \f$p\f$-th neighbor from the original point cloud
     * in the \f$i\f$-th receptive field.
     */
    std::vector<arma::Col<IndexType>> I;
    /**
     * @brief The output neighborhoods that explain how the points are
     *  connected at a given depth.
     *
     * \f[
     *  \forall 1 \leq d \leq D, \;
     *  \mathcal{N}_d \in \mathbb{Z}^{\text{bs} \times R_d \times K_d}
     * \f]
     *
     * Where \f$\text{bs}\f$ is the number of receptive fields, \f$R_d\f$ is
     * the number of points per receptive field at depth \f$d\f$, and \f$K_d\f$
     * is the number of neighbors per neighborhood at depth \f$d\f$.
     */
    std::vector<std::vector<arma::Mat<IndexType>>> N;
    /**
     * @brief The output downsampling neighborhoods that explain how the points
     *  are connected between two consecutive depths, from the first one to
     *  the second one, i.e., in the downsampling sense.
     *
     * \f[
     *  \forall 1 \leq d \leq D, \;
     *  \mathcal{N}^D_d \in \mathbb{Z}^{\text{bs} \times R_d \times K^D_d}
     * \f]
     *
     * Where \f$\text{bs}\f$ is the number of receptive fields, \f$R_d\f$ is
     *  the number of points per receptive field at depth \f$d\f$, and
     *  \f$K^D_d\f$ is the number of neighbors in the downsampling
     *  neighborhoods at depth \f$d\f$.
     */
    std::vector<std::vector<arma::Mat<IndexType>>> ND;
    /**
     * @brief The output upsampling neighborhoods that explain how the points
     *  are connected between two consecutive depths, from the second one to
     *  the first one, i.e., in the upsampling sense.
     *
     * For a hierarchy they can be expressed as follows (in the general case).
     *
     * \f[
     *  \forall 1 \leq d \leq D, \;
     *  \mathcal{N}^U_d \in \mathbb{Z}^{\text{bs} \times R_{d+1} \times K^U_d}
     * \f]
     *
     * Where \f$\text{bs}\f$ is the number of receptive fields, \f$R_{d+1}\f$
     * is the number of points per receptive field at depth \f$d+1\f$, and
     * \f$K^U_d\f$ is the number of neighbors in the upsampling
     * neighborhoods at depth \f$d\f$.
     *
     * Yet, for a non-hierarchical receptive field but also the upsampling from
     * the first receptive field to the original neighborhood, it must be
     * expressed as follows (because the number of input points \f$m_k\f$ can
     * be different for each receptive field):
     *
     * \f[
     *  \forall 1 \leq k \leq \text{bs}, \;
     *  \pmb{N}^U_d \in \mathbb{Z}^{\times m_k \times K^U}
     * \f]
     */
    std::vector<std::vector<arma::Mat<IndexType>>> NU;

    // ***   CONSTRUCTION / DESTRUCTION   *** //
    // ************************************** //
    /**
     * @brief Default constructor for DLPreProcesorOutput
     */
    DLPreProcessorOutput() = default;
    virtual ~DLPreProcessorOutput() = default;
};

}