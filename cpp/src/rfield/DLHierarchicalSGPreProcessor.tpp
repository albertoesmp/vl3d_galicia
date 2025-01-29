#include <rfield/DLHierarchicalSGPreProcessor.hpp>

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <
    typename XDecimalType,
    typename FDecimalType,
    typename IndexType,
    typename LabelType
>
DLHierarchicalSGPreProcessor<
    XDecimalType,
    FDecimalType,
    IndexType,
    LabelType
>::DLHierarchicalSGPreProcessor(
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
) :
    supportNeighborhoods(supportNeighborhoods),
    ny(ny),
    size(size),
    w(w),
    wD(wD),
    wU(wU),
    sD(sD),
    sU(sU),
    nthreads(nthreads)
{
    // Handle as many threads as available
    if(this->nthreads==-1) this->nthreads=std::thread::hardware_concurrency();
}

// ***  PRE-PROCESSING METHODS  *** //
// ******************************** //
template <
    typename XDecimalType,
    typename FDecimalType,
    typename IndexType,
    typename LabelType
>
DLSparsePreProcessorOutput<
    XDecimalType, FDecimalType, IndexType, LabelType
>
DLHierarchicalSGPreProcessor<
    XDecimalType, FDecimalType, IndexType, LabelType
>::operator()(
    arma::Mat<XDecimalType> const &Xin,
    arma::Mat<FDecimalType> const &Fin,
    arma::Col<LabelType> const &yin
){
    // Prepare receptive field computation
    IndexType const nx = Xin.n_cols; // Structure space dimensionality
    IndexType const nf = Fin.n_cols; // Feature space dimensionality
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most
    arma::Mat<XDecimalType> Xsup;
    supportNeighborhoods.computeSupportPoints(Xin, yin, Xsup);
    arma::uword const bs = Xsup.n_rows; // Number of receptive fields
    KDTree<IndexType, XDecimalType> kdt =
        supportNeighborhoods.buildKDTree(Xin);
    // Prepare output
    auto out = DLSparsePreProcessorOutput<
        XDecimalType, FDecimalType, LabelType, IndexType
    >();
    out.Fout.resize(bs);
    if(yin.n_rows > 0) out.yout.resize(bs);
    out.hsg.resize(
        bs,
        HierarchicalSparseGrid<XDecimalType, IndexType>(
            size, w, wD, wU, sD, sU, nthreads
        )
    );
    // Compute receptive fields
    int const chunkSize = MultithreadingUtils::correctChunkSize(
        bs, VL3DPP_OMP_CHUNK_SIZE_SMALL, nthreads
    );
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE, chunkSize) \
        shared(bs, Xsup, Xin, Fin, yin, kdt, nx, nf, out, chunkSize)
    for(arma::uword i = 0 ; i < bs ; ++i){ // Iterate over receptive fields
        // Compute neighborhood of support point
        arma::Col<XDecimalType> const &xi = Xsup.row(i).as_col();
        arma::Col<IndexType> Ii;
        supportNeighborhoods.computeNeighborhood(Xin, xi, kdt, Ii);
        arma::uword const mi = Ii.n_elem;
        arma::Mat<XDecimalType> Xi(mi, nx);
        arma::Mat<FDecimalType> Fi(mi, nf);
        for(arma::uword k = 0 ; k < mi ; ++k){
            Xi.row(k) = arma::conv_to<arma::Row<XDecimalType>>::from(
                Xin.row(Ii[k]).as_col()
            );
            Fi.row(k) = arma::conv_to<arma::Row<FDecimalType>>::from(
                Fin.row(Ii[k]).as_col()
            );
        }
        // Generate sparse grid representing the neighborhood
        out.hsg[i].setLogTime(false);
        out.hsg[i].fit(Xi);
        // Encode features
        // TODO Rethink : Reduce strategy from argument
        out.Fout[i] = out.hsg[i].template encodeMatrix<FDecimalType>(
            Xi, Fi, "mean"
        );
        // Encode labels, if given
        if(yin.n_rows > 0){
            arma::Col<IndexType> yi(mi);
            for(arma::uword k = 0 ; k < mi ; ++k){
                yi[k] = yin[Ii[k]];
            }
            out.yout[i] = out.hsg[i].template encodeVector<LabelType>(
                Xi, yi, "mode"
            );
        }
    }
    // Return output
    return out;
}
