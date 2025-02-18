#include <rfield/DLFPSPostProcessor.hpp>

using vl3dpp::rfield::DLFPSPostProcessor;


// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <
    typename FDecimalType,
    typename EncodingIndexType,
    typename DecodingIndexType
>
DLFPSPostProcessor<
    FDecimalType,
    EncodingIndexType,
    DecodingIndexType
>::DLFPSPostProcessor(
    std::string const &reductionType,
    FDecimalType const minClipValue,
    int const nthreads
) :
    reductionType(reductionType),
    minClipValue(minClipValue),
    nthreads(nthreads)
{
    // Handle as many threads as available
    if(this->nthreads==-1) this->nthreads=std::thread::hardware_concurrency();
    // Determine reduction type
    if(reductionType == "mean_reduce"){
        reduceFunction = &DLFPSPostProcessor::meanReduce;
    }
    else if(reductionType == "sum_reduce"){
        reduceFunction = &DLFPSPostProcessor::sumReduce;
    }
    else if(reductionType == "max_reduce"){
        reduceFunction = &DLFPSPostProcessor::maxReduce;
    }
    else if(reductionType == "entropic_reduce"){
        reduceFunction = &DLFPSPostProcessor::entropicReduce;
    }
    else{
        std::stringstream ss;
        ss  << "DLFPSPostProcessor cannot be instantiated for requested "
            << "entropic reduction strategy: \""
            << reductionType << "\"";
        throw util::VL3DPPException(ss.str());
    }
}



// ***  POST-PROCESSING METHODS  *** //
// ********************************* //
template <
    typename FDecimalType,
    typename EncodingIndexType,
    typename DecodingIndexType
>
arma::Mat<FDecimalType>
DLFPSPostProcessor<
    FDecimalType, EncodingIndexType, DecodingIndexType
>::operator()(
    arma::uword const m,
    std::vector<arma::Mat<EncodingIndexType>> const &MBatch,
    std::vector<arma::Col<DecodingIndexType>> const &I,
    arma::Cube<FDecimalType> const &zBatch
) const{
    // Prepare computation
    size_t const bs = MBatch.size(); // Batch size
    size_t const nc = zBatch.n_slices; // Number of classes
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most

    // Batch of encoded probabilities
    std::vector<arma::Mat<FDecimalType>> encoded(bs);

    // Propagate/encode probabilities
    int const chunkSize = util::MultithreadingUtils::correctChunkSize(
        bs, VL3DPP_OMP_CHUNK_SIZE_SMALL, nthreads
    );
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE, chunkSize) \
        shared(MBatch, zBatch, encoded, bs, chunkSize)
    for(size_t i = 0 ; i < bs ; ++i){
        arma::Mat<FDecimalType> const &zBatchi = zBatch.row(i);
        encoded[i] = rfield::propagateMean<FDecimalType, EncodingIndexType>(
            MBatch[i], zBatchi
        );
    }

    // Matrix of point-wise decoded probabilities
    arma::Mat<FDecimalType> decoded(m, nc, arma::fill::zeros);

    // Decode probabilities
    (this->*reduceFunction)(nc, I, encoded, decoded);

    // Return
    return decoded;
}

// ***  REDUCTION METHODS  *** //
// *************************** //
template <
    typename FDecimalType,
    typename EncodingIndexType,
    typename DecodingIndexType
>
void
DLFPSPostProcessor<
    FDecimalType, EncodingIndexType, DecodingIndexType
>::meanReduce(
    size_t const nc,
    std::vector<arma::Col<DecodingIndexType>> const &I,
    std::vector<arma::Mat<FDecimalType>> &encoded,
    arma::Mat<FDecimalType> &out
) const {
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(I, encoded, nc, out)
    for(size_t c = 0 ; c < nc ; ++c){
        rfield::decodingReduceMean<FDecimalType, DecodingIndexType>(
            I, c, encoded, out
        );
    }
}

template <
    typename FDecimalType,
    typename EncodingIndexType,
    typename DecodingIndexType
>
void
DLFPSPostProcessor<
    FDecimalType, EncodingIndexType, DecodingIndexType
>::sumReduce(
    size_t const nc,
    std::vector<arma::Col<DecodingIndexType>> const &I,
    std::vector<arma::Mat<FDecimalType>> &encoded,
    arma::Mat<FDecimalType> &out
) const {
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(I, encoded, nc, out)
    for(size_t c = 0 ; c < nc ; ++c){
        rfield::decodingReduceSum<FDecimalType, DecodingIndexType>(
            I, c, encoded, out
        );
    }
}

template <
    typename FDecimalType,
    typename EncodingIndexType,
    typename DecodingIndexType
>
void
DLFPSPostProcessor<
    FDecimalType, EncodingIndexType, DecodingIndexType
>::maxReduce(
    size_t const nc,
    std::vector<arma::Col<DecodingIndexType>> const &I,
    std::vector<arma::Mat<FDecimalType>> &encoded,
    arma::Mat<FDecimalType> &out
) const {
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(I, encoded, nc, out)
    for(size_t c = 0 ; c < nc ; ++c){
        rfield::decodingReduceMax<FDecimalType, DecodingIndexType>(
            I, c, encoded, out
        );
    }
}

template <
    typename FDecimalType,
    typename EncodingIndexType,
    typename DecodingIndexType
>
void
DLFPSPostProcessor<
    FDecimalType, EncodingIndexType, DecodingIndexType
>::entropicReduce(
    size_t const nc,
    std::vector<arma::Col<DecodingIndexType>> const &I,
    std::vector<arma::Mat<FDecimalType>> &encoded,
    arma::Mat<FDecimalType> &out
) const {
    size_t const bs = encoded.size(); // Batch size
    FDecimalType const EXP_MINUS_ONE = std::exp(-1);
    FDecimalType const Enorm = -((FDecimalType)nc) * \
        EXP_MINUS_ONE*std::log2(EXP_MINUS_ONE);
    // Clip probabilities so they are not below the threshold
    for(arma::Mat<FDecimalType> &encodei : encoded){
        FDecimalType const maxi = encodei.max();
        if(minClipValue < maxi){
            encodei = std::move(arma::clamp(
                encodei, minClipValue, encodei.max()
            ));
        }
    }
    // Compute normalized entropies
    std::vector<arma::Col<FDecimalType>> E(bs);
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(E, bs, encoded, Enorm, arma::fill::zeros)
    for(size_t t = 0 ; t < bs ; ++t){
        arma::Mat<FDecimalType> const encodet = encoded[t];
        arma::Col<FDecimalType> Et(encodet.n_rows, arma::fill::zeros);
        for(arma::uword i = 0 ; i < encodet.n_rows ; ++i){
            arma::Row<FDecimalType> const LOG2 = arma::log2(
                encodet.row(i)
            );
            for(arma::uword j = 0 ; j < encodet.n_cols ; ++j){
                Et.at(i) += encodet.at(i, j) * LOG2[j];
            }
            Et.at(i) = ((FDecimalType)1) + Et.at(i)/Enorm;
        }
        E[t] = Et;
    }
    // Compute the entropic reduction itself
    arma::Col<FDecimalType> lazyNorm(out.n_rows, arma::fill::zeros);
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(I, encoded, nc, E, out, lazyNorm)
    for(size_t c = 0 ; c < nc ; ++c){
        rfield::decodingReduceLazyWeightedMean<
            FDecimalType, DecodingIndexType
        >(
            I, c, encoded, E, out, lazyNorm
        );
    }
    // Divide by lazyNorm to achieve the weighted mean of entropies
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(out, lazyNorm)
    for(size_t i = 0 ; i < out.n_rows ; ++i){
        FDecimalType const denom = lazyNorm.at(i);
        if(denom != 0) out.row(i) /= denom;
    }
}
