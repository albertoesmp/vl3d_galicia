#ifndef VL3DPP_DL_FPS_PRE_PROCESSOR_
#define VL3DPP_DL_FPS_PRE_PROCESSOR_

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

namespace vl3dpp::rfield {


// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing a furthest point subsampling pre-processor
 *  for deep learning models.
 *
 * @tparam InputXDecimalType The data type for the decimal numbers in
 *  the context of the input structure space.
 * @tparam OutputXDecimalType The data type for the decimal numbers in
 *  the context of the output structure spaces.
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
class DLFPSPreProcessor{
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
     *  FPS pre-processor.
     */
    LabelType ny;
    /**
     * @brief Whether to transform the structure spaces of the receptive field
     *  scaling them to the unit sphere (true) or not (false).
     */
    bool toUnitSphere;
    /**
     * @brief The number of points \f$R\f$ for each FPS receptive field.
     */
    InternalIndexType R;
    /**
     * @brief The number of neighbors to consider when encoding
     *  \f$K^D\f$
     *  (i.e., dimensionality reduction).
     *
     * It is also referred to as \f$m^*\f$ in the Python documentation.
     */
    InternalIndexType KD;
    /**
     * @brief The fast method to compute the furthest point subsampling (FPS).
     *
     * <b>0</b>: No fast method, exhaustive FPS will be computed.<br/>
     * <b>1</b>: Fast method, first uniform downsampling with a given discrete
     *      step, then exhaustive FPS until ssNumPoints is reached.<br/>
     * <b>2</b>: Turo-fast method, it is purely stochastic.
     */
    short fast;
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
     * @brief Build a furthest point sampling (FPS) pre-processor for deep
     *  learning.
     * @see DLFPSPreProcessor::R
     * @see DLFPSPreProcessor::KD
     * @see DLFPSPreProcessor::fast
     * @see DLFPSPreProcessor::oversampler
     * @see vl3dpp::alg::Oversampler
     * @see DLFPSPreProcessor::nthreads
     */
    DLFPSPreProcessor(
        SupportNeighborhoods<InputXDecimalType, LabelType, InternalIndexType>
            &supportNeighborhoods,
        LabelType const ny,
        bool const toUnitSphere,
        InternalIndexType const R,
        InternalIndexType const KD,
        short const fast,
        Oversampler<OutputXDecimalType, OutputIndexType> *oversampler,
        int const nthreads
    );
    virtual ~DLFPSPreProcessor() = default;

    // ***  PRE-PROCESSING METHODS  *** //
    // ******************************** //
    /**
     * @brief Pre-process the given input point cloud
     * @param Xin The input structure space that must be pre-processed.
     * @param Fin The input feature space that must be pre-processed,
     * @param yin The input classes that must be pre-processed.
     * @return The clean receptive fields representing the input.
     * @see DLFPSPreProcessor::fit
     * @see DLFPSPreProcessor::cleanOutput
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
     * @brief Fit the indexing matrices for the points in the given receptive
     *  field. Also encode the features and labels (if available) and compute
     *  the structure space of the receptive field. After calling
     *  fit, the output and clean flags will be available.
     *
     * <b>NOTE</b> that, in general, after all receptive fields have been fit
     *  it is recommended to call DLFPSPreProcessor::cleanOutput to remove
     *  invalid receptive fields.
     *
     * This method is called from DLFPSPreProcessor::operator().
     *
     * @param[in] i The index of the receptive field to be fit.
     * @param[in] nx The dimensionality of the structure space.
     * @param[in] Xin The original input structure space.
     * @param[in] Xsup The support points derived from the original structure
     *  space.
     * @param[in] Fin The original input feature space.
     * @param[in] yin The original input labels.
     * @param[in] kdt The KDTree built on the input structure space (Xin).
     * @param[in] fps The furthest point subsampler (FPS) that must be applied
     *  to the receptive field.
     * @param[out] clean The vectors of clean labels for each receptive field.
     * @param[out] out The output where the receptive field must be stored.
     * @return Nothing, but the output is stored in place.
     * @see vl3dpp::rfield::DLFPSPreProcessor
     * @see DLFPSPreProcessor::operator()
     * @see DLFPSPreProcessor::cleanOutput
     * @see vl3dpp::rfield::DLPreProcessorOutput
     */
    void fit(
        arma::uword const i,
        InternalIndexType const nx,
        arma::Mat<InputXDecimalType> const &Xin,
        arma::Mat<InputXDecimalType> const &Xsup,
        arma::Mat<FDecimalType> const &Fin,
        arma::Col<LabelType> const &yin,
        KDTree<InternalIndexType, InputXDecimalType> &kdt,
        FurthestPointSubsampler<OutputXDecimalType> const &fps,
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
     *  DLFPSPreProcessor::operator(). Cleaning means removing all receptive
     *  fields that are not labeled as clean.
     * @param[in] clean Vector that labels each receptive field (0 means not clean,
     *  1 means clean).
     * @param[out] out The output that must be cleaned (updated inplace).
     * @see DLFPSPreProcessor::operator()
     */
    void cleanOutput(
        InternalIndexType const bs,
        arma::Mat<FDecimalType> const &Fin,
        arma::Col<LabelType> const &yin,
        arma::Col<arma::u8> const &clean,
        DLPreProcessorOutput<
            InputXDecimalType, OutputXDecimalType, FDecimalType, LabelType,
            OutputIndexType
        > &out
    );
};


#include <rfield/DLFPSPreProcessor.tpp>


}


#endif