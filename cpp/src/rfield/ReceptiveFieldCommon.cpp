// ***   INCLUDES   *** //
// ******************** //
#include <rfield/ReceptiveFieldCommon.hpp>


// ***  PROPAGATION METHODS  *** //
// ***************************** //
template <typename FDecimalType, typename IndexType>
arma::Mat<FDecimalType> propagateMean(
    arma::Mat<IndexType> const &M,
    arma::Mat<FDecimalType> const &F
){
    arma::uword const m = M.n_rows; // Number of points in original space
    arma::uword const K = M.n_cols; // Number of neighbors
    arma::uword const n = F.n_cols; // Number of features
    arma::Mat<FDecimalType> propF(m, n, arma::fill::zeros); // Output matrix
    for(arma::uword i = 0 ; i < m ; ++i){ // Iterate ith-point in orig. space
        for(arma::uword j = 0 ; j < n ; ++j){ // Iterate jth-feature
            for(arma::uword k = 0 ; k < K ; ++k){ // Iterate kth-neighbor
                propF.at(i, j) += F.at(M.at(i, k), j); // Sum
            }
        }
    }
    return propF/((FDecimalType) K);
}

template <typename FDecimalType, typename IndexType>
arma::Mat<FDecimalType> propagateClosest(
    arma::Mat<IndexType> const &M,
    arma::Mat<FDecimalType> const &F
){
    arma::uword const m = M.n_rows; // Number of points
    arma::uword const n = F.n_cols; // Number of features
    arma::Mat<FDecimalType> propF(m, n, arma::fill::zeros); // Output matrix
    for(arma::uword i = 0 ; i < m ; ++i){ // Iterate ith-point
        for(arma::uword j = 0 ; j < n ; ++j){ // Iterate jth-feature
            propF.at(i, j) += F.at(M.at(i, 0), j); // Sum
        }
    }
    return propF;
}


// ***  REDUCTION METHODS  *** //
// *************************** //
template <typename FDecimalType, typename IndexType>
arma::Mat<FDecimalType> reduceMean(
    arma::Mat<IndexType> const &N,
    arma::Mat<FDecimalType> const &F
){
    arma::uword const R = N.n_rows; // Number of points in receptive field
    arma::uword const K = N.n_cols; // Number of neighbors
    arma::uword const n = F.n_cols; // Number of features
    arma::Mat<FDecimalType> redF(R, n, arma::fill::zeros); // Output matrix
    for(arma::uword i = 0 ; i < R ; ++i){ // Iterate ith-point in recep. field
        for(arma::uword j = 0 ; j < n ; ++j){ // Iterate jth-feature
            for(arma::uword k = 0 ; k < K ; ++k){ // Iterate kth-neighbor
                redF.at(i, j) += F.at(N.at(i, k), j); // Sum
            }
        }
    }
    return redF/((FDecimalType) K);
}

template <typename LabelType, typename IndexType>
arma::Col<LabelType> reduceLabelMode(
    arma::Mat<IndexType> const &N,
    arma::Col<LabelType> const &y,
    LabelType const ny
){
    IndexType const R = N.n_rows; // Number of points in receptive field
    IndexType const K = N.n_cols; // Number of neighbors
    std::vector<LabelType> redy(R); // Output vector of labels
    std::vector<IndexType> count(ny); // Count pts/class
    for(IndexType i = 0 ; i < R ; ++i){ // Iterate i-th point in recep. field
        // Initialize count to zeros
        std::memset(count.data(), 0, count.size() * sizeof(IndexType));
        // Point-wise class recount in the neighborhood
        for(IndexType k = 0 ; k < K ; ++k){ // Iterate k-th neighbor
            count[y[N.at(i, k)]] += 1;
        }
        // Select modal class
        IndexType maxCount = count[0];
        LabelType modeClass = 0;
        for(LabelType j = 1 ; j < ny ; ++j){ // Iterate j-th class
            if(count[j] > maxCount){ // Check count is greater for j-th class
                maxCount = count[j];
                modeClass = j;
            }
        }
        redy[i] = modeClass; // Assign mode class for i-th point in recep. field
    }
    // Index of max count gives the class
    return arma::conv_to<arma::Col<LabelType>>::from(redy);
}

template <typename LabelType, typename IndexType>
arma::Col<LabelType> reduceLabelClosest(
    arma::Col<IndexType> const &N,
    arma::Col<LabelType> const &y,
    LabelType const ny
){
    arma::uword const R = N.n_rows; // Number of points in receptive field
    arma::uword const K = N.n_cols; // Number of neighbors
    arma::Col<LabelType> redy(R); // Output vector of labels
    for(arma::uword i = 0 ; i < R ; ++i){ // Iterate ith-point in recep. field
        redy[i] = y[N[i]]; // Index of max count gives the class
    }
    return redy;
}


// ***  DECODING REDUCTION METHODS  *** //
// ************************************ //
template <typename FDecimalType, typename IndexType>
void decodingReduceMean(
    std::vector<arma::Col<IndexType>> const &I,
    arma::uword const outColIdx,
    std::vector<arma::Mat<FDecimalType>> const &encoded,
    arma::Mat<FDecimalType> &out
){
    // Prepare
    size_t const bs = encoded.size(); // Batch size
    arma::Col<IndexType> count(out.n_rows); // Recount
    // Sum
    for(size_t i = 0 ; i < bs ; ++i){ // i-th element in batch
        arma::Col<IndexType> const &Ii = I[i]; // i-th neighborhood
        arma::Mat<FDecimalType> const &encodedi = encoded[i]; // i-th encoding
        for(size_t k = 0 ; k < Ii.n_rows ; ++k){
            arma::uword const ik = Ii[k];
            out.at(ik, outColIdx) += encodedi.at(k, outColIdx);
            count.at(ik) += 1;
        }
    }
    // Divide by count to obtain the mean
    for(size_t i = 0 ; i < out.n_rows ; ++i){
        IndexType const counti = count.at(i);
        if(counti != 0) out.at(i, outColIdx) /= (FDecimalType) counti;
    }
}

template <typename FDecimalType, typename IndexType>
void decodingReduceSum(
    std::vector<arma::Col<IndexType>> const &I,
    arma::uword const outColIdx,
    std::vector<arma::Mat<FDecimalType>> const &encoded,
    arma::Mat<FDecimalType> &out
){
    // Prepare
    size_t const bs = encoded.size(); // Batch size
    // Sum
    for(size_t i = 0 ; i < bs ; ++i){ // i-th element in batch
        arma::Col<IndexType> const &Ii = I[i]; // i-th neighborhood
        arma::Mat<FDecimalType> const &encodedi = encoded[i]; // i-th encoding
        for(size_t k = 0 ; k < Ii.n_rows ; ++k){
            arma::uword const ik = Ii[k];
            out.at(ik, outColIdx) += encodedi.at(k, outColIdx);
        }
    }
}

template <typename FDecimalType, typename IndexType>
void decodingReduceMax(
    std::vector<arma::Col<IndexType>> const &I,
    arma::uword const outColIdx,
    std::vector<arma::Mat<FDecimalType>> const &encoded,
    arma::Mat<FDecimalType> &out
){
    // Prepare
    size_t const bs = encoded.size(); // Batch size
    // Sum
    for(size_t i = 0 ; i < bs ; ++i){ // i-th element in batch
        arma::Col<IndexType> const &Ii = I[i]; // i-th neighborhood
        arma::Mat<FDecimalType> const &encodedi = encoded[i]; // i-th encoding
        for(size_t k = 0 ; k < Ii.n_rows ; ++k){
            arma::uword const ik = Ii[k];
            FDecimalType const zikj = out.at(ik, outColIdx);
            FDecimalType const eikj = encodedi.at(k, outColIdx);
            if(eikj > zikj) out.at(ik, outColIdx) = eikj;
        }
    }
}

template <typename FDecimalType, typename IndexType>
void decodingReduceLazyWeightedMean(
    std::vector<arma::Col<IndexType>> const &I,
    arma::uword const outColIdx,
    std::vector<arma::Mat<FDecimalType>> const &encoded,
    std::vector<arma::Col<FDecimalType>> const &W,
    arma::Mat<FDecimalType> &out,
    arma::Col<FDecimalType> &lazyNorm
){
    // Prepare
    size_t const bs = encoded.size(); // Batch size
    // Lazy weighted sum
    for(size_t i = 0 ; i < bs ; ++i){ // i-th element in batch
        arma::Col<IndexType> const &Ii = I[i]; // i-th neighborhood
        arma::Mat<FDecimalType> const &encodedi = encoded[i]; // i-th encoding
        arma::Col<FDecimalType> const &wi = W[i];
        for(size_t k = 0 ; k < Ii.n_rows ; ++k){
            arma::uword const ik = Ii[k];
            FDecimalType const wik = wi[k];
            out.at(ik, outColIdx) += wik * encodedi.at(k, outColIdx);
            if(outColIdx == 0) lazyNorm.at(ik) += wik;
        }
    }
    // Note that dividing by the lazy norm must be done outside this function
    // scope. This is mostly used for parallel computations.
}


// ***  TRANSFORMATIONS  *** //
// ************************* //
template <typename XDecimalType>
void toUnitSphere(arma::Mat<XDecimalType> &X){
    XDecimalType const r = std::sqrt(arma::max(arma::sum(arma::square(X), 1)));
    X /= r;
}
