#ifndef VL3DPP_DL_HIERARCHICAL_SG_PRE_PROCESSOR_
#define VL3DPP_DL_HIERARCHICAL_SG_PRE_PROCESSOR_

// ***   INCLUDES   *** //
// ******************** //
#include <rfield/DLSparsePreProcessorOutput.hpp>
#include <adt/grid/HierarchicalSparseGrid.hpp>
#include <alg/SupportNeighborhoods.hpp>
#include <util/VL3DPPException.hpp>
#include <util/VL3DPPMacros.hpp>
#include <util/MultithreadingUtils.hpp>

#include <armadillo>
#include <omp.h>

#include <thread>
#include <string>
#include <vector>
#include <sstream>

using vl3dpp::rfield::DLSparsePreProcessorOutput;
using vl3dpp::alg::SupportNeighborhoods;
using vl3dpp::adt::grid::HierarchicalSparseGrid;
using vl3dpp::util::VL3DPPException;
using vl3dpp::util::MultithreadingUtils;


namespace vl3dpp::rfield{

// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing a hierarchical sparse grid pre-processor for deep
 *  learning models.
 *
 * @tparam XDecimalType The data type for the decimal numbers in the
 *  context of the input structure space.
 * @tparam FDecimalType The data type for the decimal numbers in the context
 *  of the feature space.
 * @tparam IndexType The index type for integers representing cells or
 *  active cells.
 * @tparam LabelType The data ype used to codify point-wise classes (labels)
 */
template <
    typename XDecimalType,
    typename FDecimalType,
    typename IndexType,
    typename LabelType
>
class DLHierarchicalSGPreProcessor{
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief The vl3dpp::alg::SupportNeighborhoods object governing how the
     *  pre-processor computes the support points.
     * @see vl3dpp::alg::SupportNeighborhoods
     */
    SupportNeighborhoods<XDecimalType, LabelType, IndexType>
        &supportNeighborhoods;
    /**
     * @brief The number of different classes that must be supported by the
     *   hierarchical sparse grid pre-processor.
     */
    LabelType ny;
    /**
     * @see HierarchicalSparseGrid::sg
     * @see SparseGrid::size
     */
    XDecimalType size;
    /**
     * @see HierarchicalSparseGrid::w
     */
    arma::Col<IndexType> w;
    /**
     * @see HierarchicalSparseGrid::wD
     */
    arma::Col<IndexType> wD;
    /**
     * @see HierarchicalSparseGrid::wU
     */
    arma::Col<IndexType> wU;
    /**
     * @see HierarchicalSparseGrid::sD
     */
    arma::Col<IndexType> sD;
    /**
     * @see HierarchicalSparseGrid::sU
     */
    arma::Col<IndexType> sU;
    /**
     * @brief How many threads must be used.
     *
     * If 0, then the number of threads depends on the current configuration.
     * If -1, then all available threads will be used.
     */
    int nthreads;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build a hierarchical furthest point sampling (HFPS) pre-processor
     *  for deep learning.
     * @see DLHierarchicalSGPreProcessor::supportNeighborhoods
     * @see DLHierarchicalSGPreProcessor::ny
     * @see DLHierarchicalSGPreProcessor::size
     * @see DLHierarchicalSGPreProcessor::w
     * @see DLHierarchicalSGPreProcessor::wD
     * @see DLHierarchicalSGPreProcessor::wU
     * @see DLHierarchicalSGPreProcessor::sD
     * @see DLHierarchicalSGPreProcessor::sU
     * @see DLHierarchicalSGPreProcessor::nthreads
     */
    DLHierarchicalSGPreProcessor(
        SupportNeighborhoods<XDecimalType, LabelType, IndexType>
            &supportNeighborhoods,
        LabelType const ny,
        XDecimalType const size,
        arma::Col<IndexType> const &w,
        arma::Col<IndexType> const &wD,
        arma::Col<IndexType> const &wU,
        arma::Col<IndexType> const &sD,
        arma::Col<IndexType> const &sU,
        int const nthreads
    );
    virtual ~DLHierarchicalSGPreProcessor() = default;

    // ***  PRE-PROCESSING METHODS  *** //
    // ******************************** //
    /**
     * @brief Pre-process the given input point cloud.
     * @param Xin The input structure space that must be pre-processed.
     * @param Xin The input structure space that must be pre-processed.
     * @param Fin The input feature space that must be pre-processed,
     * @param yin The input classes that must be pre-processed.
     * @return The clean receptive fields representing the input.
     */
    DLSparsePreProcessorOutput<
        XDecimalType, FDecimalType, IndexType, LabelType
    > operator()(
        arma::Mat<XDecimalType> const &Xin,
        arma::Mat<FDecimalType> const &Fin,
        arma::Col<LabelType> const &yin
    );
};

#include <rfield/DLHierarchicalSGPreProcessor.tpp>

}

#endif
