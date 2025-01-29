#ifndef VL3DPP_DL_HIERARCHICAL_FPS_PRE_PROCESSOR_
#define VL3DPP_DL_HIERARCHICAL_FPS_PRE_PROCESSOR_

// ***   INCLUDES   *** //
// ******************** //
#include <rfield/ReceptiveFieldCommon.hpp>
#include <rfield/DLPreProcessorOutput.hpp>
#include <alg/FurthestPointSubsampler.hpp>
#include <alg/SupportNeighborhoods.hpp>
#include <alg/Oversampler.hpp>
#include <util/VL3DPPException.hpp>
#include <util/VL3DPPMacros.hpp>
#include <util/MultithreadingUtils.hpp>

#include <armadillo>
#include <omp.h>

#include <thread>
#include <string>
#include <vector>
#include <sstream>

using vl3dpp::alg::Oversampler;
using vl3dpp::alg::SupportNeighborhoods;
using vl3dpp::alg::FurthestPointSubsampler;
using vl3dpp::util::VL3DPPException;
using vl3dpp::util::MultithreadingUtils;

namespace vl3dpp::rfield{


// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing a hierarchical furthest point subsampling
 *  pre-processor for deep learning models.
 *
 * @tparam InputXDecimalType The data type for the decimal numbers in the
 *  context of the input structure space.
 * @tparam OutputXDecimalType The data type for the decimal numbers in the
 *  contexts of the output structure spaces.
 * @tparam FDecimalType The data type for the decimal numbers in the context
 *  of the feature space.
 * @tparam InternalIndexType The index type to be used internally by the
 *  pre-processor.
 * @tparam OutputIndexType The index type of the output neighborhoods.
 * @tparam LabelType The data type used to codify point-wise classes (labels).
 */
template <
    typename InputXDecimalType,
    typename OutputXDecimalType,
    typename FDecimalType,
    typename InternalIndexType,
    typename OutputIndexType,
    typename LabelType
>
class DLHierarchicalFPSPreProcessor{
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief The vl3dpp::alg::SupportNeighborhoods object governing how the
     *  pre-processor computes the support points.
     * @see vl3dpp::alg::SupportNeighborhoods
     */
    SupportNeighborhoods<InputXDecimalType, LabelType, InternalIndexType>
        &supportNeighborhoods;
    /**
     * @brief The number of different classes that must be supported by the
     *  hierarchical FPS pre-processor.
     */
    LabelType ny;
    /**
     * @brief Whether to transform the structure spaces of the receptive field
     *  scaling them to the unit sphere (true) or not (false).
     */
    bool toUnitSphere;
    /**
     * @brief The number of points \f$R_d\f$ at each depth \f$1 < d < d^*\f$
     *  for each FPS receptive field.
     */
    arma::Col<InternalIndexType> R;
    /**
     * @brief The number of neighbors to consider when encoding
     *  \f$K^D_d\f$ at depth \f$1 < d < d^*\f$
     *  (i.e., dimensionality reduction).
     */
    arma::Col<InternalIndexType> KD;
    /**
     * @brief The number of neighbors to consider when decoding
     *  \f$K^U_d\f$ at depth \f$1 < d < d^*\f$
     *  (i.e., dimensionality augmentation/restoration).
     */
    arma::Col<InternalIndexType> KU;
    /**
     * @brief The number of neighbors to consider for neighborhoods at the same
     *  depth level \f$K_d\f$ with \f$1 < d < d^*\f$.
     */
    arma::Col<InternalIndexType> KN;
    /**
     * @brief The fast method to compute the furthest point subsampling (FPS)
     *  at each depth \f$1 < d < d^*\f$.
     *
     * <b>0</b>: No fast method, exhaustive FPS will be computed.<br/>
     * <b>1</b>: Fast method, first uniform downsampling with a given discrete
     *      step, then exhaustive FPS until ssNumPoints is reached.<br/>
     * <b>2</b>: Turo-fast method, it is purely stochastic.
     */
    std::vector<short> fast;
    /**
     * @brief Pointer to the vl3dpp::alg::oversampler object that applies the
     *  requested oversampling. It will be a null pointer if no oversampling is
     *  requested.
     * @see vl3dpp::alg::Oversampler
     */
    Oversampler<OutputXDecimalType, OutputIndexType> *oversampler;
    /**
     * @brief How many threads must be used.
     *
     * If 0, then the number of threads depends on the current configuration.
     * If -1, then all available threads will be used.
     */
    int nthreads;

public:
    // ***  CONSTRUCTION / DESTRUCTION *** //
    // *********************************** //
    /**
     * @brief Build a hierarchical furthest point sampling (HFPS) pre-processor
     *  for deep learning.
     * @see DLHierarchicalFPSPreProcessor::R
     * @see DLHierarchicalFPSPreProcessor::KD
     * @see DLHierarchicalFPSPreProcessor::KU
     * @see DLHierarchicalFPSPreProcessor::KN
     * @see DLHierarchicalFPSPreProcessor::fast
     * @see DLHierarchicalFPSPreProcessor::oversampler
     * @see vl3dpp::alg::Oversampler
     * @see DLHierarchicalFPSPreProcessor::nthreads
     */
    DLHierarchicalFPSPreProcessor(
        SupportNeighborhoods<InputXDecimalType, LabelType, InternalIndexType>
            &supportNeighborhoods,
        LabelType const ny,
        bool const toUnitSphere,
        arma::Col<InternalIndexType> const &R,
        arma::Col<InternalIndexType> const &KD,
        arma::Col<InternalIndexType> const &KU,
        arma::Col<InternalIndexType> const &KN,
        std::vector<short> const &fast,
        Oversampler<OutputXDecimalType, OutputIndexType> *oversampler,
        int const nthreads
    );
    virtual ~DLHierarchicalFPSPreProcessor() = default;

    // ***  PRE-PROCESSING METHODS  *** //
    // ******************************** //
    /**
     * @brief Pre-process the given input point cloud.
     * @param Xin The input structure space that must be pre-processed.
     * @param Fin The input feature space that must be pre-processed,
     * @param yin The input classes that must be pre-processed.
     * @return The clean receptive fields representing the input.
     * @see DLHierarchicalFPSPreProcessor::fit
     * @see DLHierarchicalFPSPreProcessor::cleanOutput
     */
    DLPreProcessorOutput<
        InputXDecimalType, OutputXDecimalType, FDecimalType,
        LabelType, OutputIndexType
    > operator()(
        arma::Mat<InputXDecimalType> const &Xin,
        arma::Mat<FDecimalType> const &Fin,
        arma::Col<LabelType> const &yin
    );
    /**
     * @brief Fit the indexing matrices for the given receptive field. Also
     *  encode the features and labels (if available) and compute the
     *  hierarchical structure spaces. After calling fit, the output and
     *  clean flags will be available.
     *
     * @param[in] i The index of the receptive field to be fit.
     * @param[in] maxDepth The max depth for the hierarchy.
     * @param[in] nx The dimensionality of the structure space.
     * @param[in] Xin The original input structure space.
     * @param[in] Xsup The support points derived from the original structure
     *  space.
     * @param[in] Fin The original input feature space.
     * @param[in] yin The original input labels.
     * @param[in] kdt The KDTree built on the input structure space (Xin).
     * @param[in] hfps The furthest point subsampler (FPS) for each depth of
     *  the hierarchy.
     * @param[out] clean The vectors of clean labels for each receptive field.
     * @param[out] out The output where the receptive field must be stored.
     * @return Nothing, but the output is stored in place.
     * @see vl3dpp::rfield::DLFPSPreProcessor
     * @see DLHierarchicalFPSPreProcessor::operator()
     * @see DLHierarchicalFPSPreProcessor::cleanOutput
     * @see vl3dpp::rfield::DLPreProcessorOutput
     */
    void fit(
        arma::uword const i,
        InternalIndexType const maxDepth,
        InternalIndexType const nx,
        arma::Mat<InputXDecimalType> const &Xin,
        arma::Mat<InputXDecimalType> const &Xsup,
        arma::Mat<FDecimalType> const &Fin,
        arma::Col<LabelType> const &yin,
        KDTree<InternalIndexType, InputXDecimalType> &kdt,
        std::vector<FurthestPointSubsampler<OutputXDecimalType>> const &hfps,
        arma::Col<arma::u8> &clean,
        DLPreProcessorOutput<
            InputXDecimalType, OutputXDecimalType, FDecimalType,
            LabelType, OutputIndexType
        > &out
    );

    // ***  UTIL METHODS  *** //
    // ********************** //
    /**
     * @brief Clean the output generated by a call to
     *  DLHierarchicalFPSPreProcessor::operator(). Cleaning means removing all
     *  receptive fields that are not labeled as clean.
     * @param[in] clean Vector that labels each receptive field (0 means not
     *  clean, 1 means clean).
     * @param[out] out The output that must be cleaned (updated inplace).
     * @see DLHierarchicalFPSPreProcessor::operator()
     */
    void cleanOutput(
        InternalIndexType const bs,
        InternalIndexType const maxDepth,
        arma::Mat<FDecimalType> const &Fin,
        arma::Col<LabelType> const &yin,
        arma::Col<arma::u8> const &clean,
        DLPreProcessorOutput<
            InputXDecimalType, OutputXDecimalType, FDecimalType, LabelType,
            OutputIndexType
        > &out
    );
};


#include <rfield/DLHierarchicalFPSPreProcessor.tpp>


}

#endif
