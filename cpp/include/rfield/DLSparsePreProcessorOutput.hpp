#pragma once

// ***   INCLUDES   *** //
// ******************** //
#include <adt/grid/HierarchicalSparseGrid.hpp>

#include <armadillo>

#include <vector>

using vl3dpp::adt::grid::HierarchicalSparseGrid;

namespace vl3dpp::rfield{

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing the output of a deep learning pre-processor.
 *
 * @tparam InputXDecimalType The data type for the decimal numbers representing
 *  the input structure spaces (typically used to store the centers of each
 *  receptive field with the same decimal precision than the original input).
 * @tparam FDecimalType The data type for the decimal nubmers representing the
 *  featur spaces.
 * @tparam LabelType The data type for the values representing the point-wise
 *  labels (e.g., point-wise classes).
 * @tparam IndexType The index type used to encode the neighborhoods.
 *
 * @see vl3dpp::rfield::DLHierarchicalSGPreProcessor
 */
template <
    typename InputXDecimalType,
    typename FDecimalType,
    typename LabelType,
    typename IndexType
>
class DLSparsePreProcessorOutput{
public:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief The output batch of feature spaces.
     *
     * The batch itself can be seen as a tuple of matrices such that:
     * \f[
     *  F = (\ldots, \pmb{F_i} \in \mathbb{R}^{m_i \times n_f}, \ldots)
     * \ftpp]
     *
     * Where \f$m_i\f$ is the number of active cells for the \f$i\f$-th element
     * of the batch and \f$n_f\f$ is the dimensionality of the feature space
     * (i.e., the number of features).
     */
    std::vector<arma::Mat<FDecimalType>> Fout;
    /**
     * @brief The output batch of labels.
     *
     * The batch itself cna be seen as a tuple of vectors such that:
     * \f[
     *  Y = (\ldots, \pmb{y_i} \in \mathbb{Z}^{m_i}_{\geq 0}, \ldots)
     * \f]
     *
     * Where \f$m_i\f$ is the number of active cells for the \f$i\f$-th element
     * of the batch and for any \f$j\f$,
     * \f$ 0 \leq (\pmb{y_i})_j < n_y\f$
     * (with \f$n_y\f$ being the number of classes).
     */
    std::vector<arma::Col<LabelType>> yout;
    /**
     * @brief The output hierarchical sparse grids representing the input point
     *  cloud that was pre-processed.
     * @see vl3dpp::adt::grid::HierarchicalSparseGrid
     */
    std::vector<HierarchicalSparseGrid<InputXDecimalType, IndexType>> hsg;

    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Default constructor for DLSparsePreProcessorOutput
     */
    DLSparsePreProcessorOutput() = default;
    virtual ~DLSparsePreProcessorOutput() = default;

};

}