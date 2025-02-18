#ifndef VL3DPP_DL_FPS_POST_PROCESSOR_
#define VL3DPP_DL_FPS_POST_PROCESSOR_

// ***   INCLUDES   *** //
// ******************** //
#include <rfield/ReceptiveFieldCommon.hpp>
#include <util/VL3DPPException.hpp>
#include <util/VL3DPPMacros.hpp>
#include <util/MultithreadingUtils.hpp>

#include <thread>
#include <armadillo>
#include <omp.h>

#include <vector>
#include <string>
#include <sstream>
#include <functional>


namespace vl3dpp::rfield {

// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing a furthest point subsampling post-processor
 *  for deep learning models.
 *
 * @tparam FDecimalType The data type for the decimal numbers in the context
 *  of the batch of probabilities.
 * @tparam EncodingIndexType The index type for the integer numbers in the
 *  context of the propagation to the receptive fields (encoding).
 * @tparam DecodingIndexType The index type for the integer numbers in the
 *  context of the reduction from the receptive fields (decoding).
 */
template <
    typename FDecimalType,
    typename EncodingIndexType,
    typename DecodingIndexType
>
class DLFPSPostProcessor {
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief The type of reduction strategy to be used by the post-processor.
     * Supported types are:
     *
     * <ol>
     *  <li>"mean_reduce"</li>
     *  <li>"sum_reduce"</li>
     *  <li>"max_reduce"</li>
     *  <li>"entropic_reduce"</li>
     * </ol>
     *
     * @see DLFPSPostProcessor::reduceFunction
     */
    std::string reductionType;

    /**
     * @brief Some computations (e.g., entropic reduction) might require a
     *  min clip value so any value that is below this threshold is replaced
     *  by the threshold itself (e.g., to avoid computing \f$log(0)\f$).
     */
    FDecimalType minClipValue;

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
     * @brief The function to reduce the probabilities propagated to the
     *  receptive fields back to the original space.
     * @see DLFPSPostProcessor::reductionType
     * @see ReceptiveFieldCommon::decodingReduceMean
     * @see ReceptiveFieldCommon::decodingReduceSum
     * @see ReceptiveFieldCommon::decodingReduceMax
     * @see ReceptiveFieldCommon::decodingReduceLazyWeightedMean
     */
    void (DLFPSPostProcessor::*reduceFunction) (
        size_t const nc,
        std::vector<arma::Col<DecodingIndexType>> const &I,
        std::vector<arma::Mat<FDecimalType>> &encoded,
        arma::Mat<FDecimalType> &out
    ) const;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build a furthest point sampling (FPS) post-processor for deep
     *  learning.
     * @see DLFPSPostProcessor::reductionType
     * @see DLFPSPostProcessor::nthreads
     */
    DLFPSPostProcessor(
        std::string const &reductionType,
        FDecimalType const minClipValue,
        int const nthreads
    );
    virtual ~DLFPSPostProcessor() = default;

    // ***  POST-PROCESSING METHODS  *** //
    // ********************************* //
    /**
     * @brief Post-process the given batch of probabilities in the receptive
     *  fields back to the \f$m \in \mathbb{Z}_{>0}\f$ points in the original
     *  space.
     *
     * @param m The number of points in the original space.
     * @param MBatch The indices representing the encoding neighborhoods.
     * @param I The indices representing the decoding neighborhoods.
     * @param zBatch The batch representing the probabilities in the receptive
     *  fields.
     * @return The point-wise probabilities in the original space, i.e., outside
     *  the receptive fields.
     *
     * @see ReceptiveFieldCommon::decodingReduceMean
     * @see ReceptiveFieldCommon::decodingReduceSum
     * @see ReceptiveFieldCommon::decodingReduceMax
     * @see ReceptiveFieldCommon::decodingReduceLazyWeightedMean
     */
    arma::Mat<FDecimalType> operator()(
        arma::uword const m,
        std::vector<arma::Mat<EncodingIndexType>> const &MBatch,
        std::vector<arma::Col<DecodingIndexType>> const &I,
        arma::Cube<FDecimalType> const &zBatch
    ) const;

    // ***  REDUCTION METHODS  *** //
    // *************************** //
    /**
     * @brief Reduce the encoded probabilities by taking the mean.
     * @see DLFPSPostProcessor::reduceFunction
     * @see ReceptiveFieldCommon::decodingReduceMean
     */
    void meanReduce(
        size_t const nc,
        std::vector<arma::Col<DecodingIndexType>> const &I,
        std::vector<arma::Mat<FDecimalType>> &encoded,
        arma::Mat<FDecimalType> &out
    ) const;
    /**
     * @brief Reduce the encoded probabilities by summing them.
     * @see DLFPSPostProcessor::reduceFunction
     * @see ReceptiveFieldCommon::decodingReduceSum
     */
    void sumReduce(
        size_t const nc,
        std::vector<arma::Col<DecodingIndexType>> const &I,
        std::vector<arma::Mat<FDecimalType>> &encoded,
        arma::Mat<FDecimalType> &out
    ) const;
    /**
     * @brief Reduce the encoded probabilities by taking the max.
     * @see DLFPSPostProcessor::reduceFunction
     * @see ReceptiveFieldCommon::decodingReduceMax
     */
    void maxReduce(
        size_t const nc,
        std::vector<arma::Col<DecodingIndexType>> const &I,
        std::vector<arma::Mat<FDecimalType>> &encoded,
        arma::Mat<FDecimalType> &out
    ) const;
    /**
     * @brief Reduce the encoded probabilities weighting with the entropy.
     * @see DLFPSPostProcessor::reduceFunction
     * @see ReceptiveFieldCommon::decodingReduceLazyWeightedMean
     */
    void entropicReduce(
        size_t const nc,
        std::vector<arma::Col<DecodingIndexType>> const &I,
        std::vector<arma::Mat<FDecimalType>> &encoded,
        arma::Mat<FDecimalType> &out
    ) const;

};

#include <rfield/DLFPSPostProcessor.tpp>

}

#endif
