#include <rfield/DLSGPostProcessor.hpp>

using vl3dpp::rfield::DLSGPostProcessor;

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <
    typename XDecimalType,
    typename FDecimalType,
    typename IndexType
>
DLSGPostProcessor<
    XDecimalType,
    FDecimalType,
    IndexType
>::DLSGPostProcessor(
    XDecimalType const cellSize,
    vector<arma::Row<XDecimalType>> const &A,
    vector<arma::Col<IndexType>> const &n,
    vector<map<IndexType, IndexType>> const &h,
    string const &reductionType,
    FDecimalType const minClipValue,
    int const nthreads
) :
    reductionType(reductionType),
    minClipValue(minClipValue),
    cellSize(cellSize),
    A(A),
    n(n),
    h(h),
    nthreads(nthreads)
{
    // Handle as many threads as available
    if(this->nthreads==-1) this->nthreads=std::thread::hardware_concurrency();
    // Determine reduction type
    if(reductionType == "mean_reduce"){
        // Empty because reduceFunction is not implemented yet
    }
    else if(reductionType == "sum_reduce"){
        throw util::VL3DPPException(
            "Sum reduction is currently NOT supported for DLSGPostProcessor."
        );
    }
    else if(reductionType == "max_reduce"){
        throw util::VL3DPPException(
            "Max reduction is currently NOT supported for DLSGPostProcessor."
        );
    }
    else if(reductionType == "entropic_reduce"){
        throw util::VL3DPPException(
            "Entropic reduction is currently NOT supported for "
            "DLSGPostProcessor."
        );
    }
    else{
        std::stringstream ss;
        ss  << "DLSGPostProcessor cannot be instantiated for requested "
            << "entropic reduction strategy: \""
            << reductionType << "\"";
        throw util::VL3DPPException(ss.str());
    }
}

// ***  POST-PROCESSING METHODS  *** //
// ********************************* //
template <
    typename XDecimalType,
    typename FDecimalType,
    typename IndexType
>
arma::Mat<FDecimalType>
DLSGPostProcessor<
    XDecimalType,
    FDecimalType,
    IndexType
>::operator()(
    arma::Mat<XDecimalType> const &X,
    std::vector<arma::Mat<FDecimalType>> const &zBatch
) const{
    // Prepare computation
    arma::uword const m = X.n_rows; // Number of points in the original pcloud
    arma::uword const nx = X.n_cols; // Structure space dimensionality
    size_t const bs = zBatch.size(); // Number of elements in the batch
    size_t const ny = zBatch[0].n_cols; // Number of classes
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most
    arma::Col<IndexType> cardinals( // How many times a point is predicted
        m, arma::fill::zeros
    );
    arma::Mat<FDecimalType> Z( // Point-wise probabilities
        m, ny, arma::fill::zeros
    );

    // Compute point-wise probabilities from receptive field probabilities
    int const chunkSize = MultithreadingUtils::correctChunkSize(
        m, VL3DPP_OMP_CHUNK_SIZE, nthreads
    );
    for(size_t k = 0 ; k < bs ; ++k){
        map<IndexType, IndexType> const &hk = h[k];
        arma::Mat<FDecimalType> const &zk = zBatch[k];
        arma::Row<XDecimalType> const &Ak = A[k];
        SparseGrid<XDecimalType, IndexType> sgk(cellSize, Ak, n[k]);
        sgk.setLogTime(false);
        arma::Row<XDecimalType> const Bk = Ak + arma::conv_to<
            arma::Row<XDecimalType>
        >::from(n[k]) * cellSize;
        #pragma omp parallel for default(none) \
            schedule(VL3DPP_OMP_SCHEDULE, chunkSize) \
            shared(X, m, sgk, hk, Z, zk, cardinals, chunkSize, Ak, Bk, nx)
        for(arma::uword i = 0 ; i < m ; ++i){
            // TODO Rethink : Faster if ignoring points outside SG[0] bbox?
            // TODO Rethink : Alternative to solve out of bbox indexing bug ---
            // Skip points outside bounding box of first sparse grid
            bool outside = false;
            for(arma::uword j = 0 ; j < nx ; ++j){
                outside |= X.at(i, j) < Ak[j] || X.at(i, j) > Bk[j];
            }
            if(outside) continue;
            // --- TODO Rethink : Alternative to solve out of bbox indexing bug
            // Find sequential active cell index from cell index
            IndexType const idx = sgk.indexFromCoordinates(X.row(i));
            typename map<IndexType, IndexType>::const_iterator hki = hk.find(
                idx
            );
            if(hki == hk.cend()) continue; // Skip iter. if idx is not active
            // Sum probabilities
            IndexType const seqIdx = hki->second;
            Z.row(i) += zk.row(seqIdx);
            cardinals[i] += 1; // Increment corresponding cardinality counter
        }
    }
    for(arma::uword i = 0 ; i < m ; ++i){ // Divide by cardinals to compute mean
        if(cardinals[i] == 0) continue; // Skip null cardinals to avoid zero-div
        FDecimalType const cardinali = cardinals[i];
        Z.row(i) /= cardinali;
    }

    // Return point-wise probabilities
    return Z;
}