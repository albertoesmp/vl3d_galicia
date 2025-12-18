#include <rfield/DLFPSPreProcessor.hpp>


// ***  CONSTRUCTION / DESTRUCTION *** //
// *********************************** //
template <
    typename InputXDecimalType,
    typename OutputXDecimalType,
    typename FDecimalType,
    typename InternalIndexType,
    typename OutputIndexType,
    typename LabelType
>
DLFPSPreProcessor<
    InputXDecimalType,
    OutputXDecimalType,
    FDecimalType,
    InternalIndexType,
    OutputIndexType,
    LabelType
>::DLFPSPreProcessor(
    SupportNeighborhoods<InputXDecimalType, LabelType, InternalIndexType>
        &supportNeighborhoods,
    LabelType const ny,
    bool const toUnitSphere,
    InternalIndexType const R,
    InternalIndexType const KD,
    short const fast,
    Oversampler<OutputXDecimalType, OutputIndexType> *oversampler,
    int const nthreads
) :
    supportNeighborhoods(supportNeighborhoods),
    ny(ny),
    toUnitSphere(toUnitSphere),
    R(R),
    KD(KD),
    fast(fast),
    oversampler(oversampler),
    nthreads(nthreads)
{
    // Handle as many threads as available
    if(this->nthreads==-1) this->nthreads=std::thread::hardware_concurrency();
}

// ***  PRE-PROCESSING METHODS  *** //
// ******************************** //
template <
    typename InputXDecimalType,
    typename OutputXDecimalType,
    typename FDecimalType,
    typename InternalIndexType,
    typename OutputIndexType,
    typename LabelType
>
DLPreProcessorOutput<
    InputXDecimalType, OutputXDecimalType, FDecimalType,
    LabelType, OutputIndexType
>
DLFPSPreProcessor<
    InputXDecimalType,
    OutputXDecimalType,
    FDecimalType,
    InternalIndexType,
    OutputIndexType,
    LabelType
>::operator()(
    arma::Mat<InputXDecimalType> const &Xin,
    arma::Mat<FDecimalType> const &Fin,
    arma::Col<LabelType> const &yin
){
    // Prepare receptive field computation
    InternalIndexType const nx = Xin.n_cols; // Structure space dimensionality
    FurthestPointSubsampler<OutputXDecimalType> fps(R, 1);
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most
    // Compute support points and KDTree
    arma::Mat<InputXDecimalType> Xsup;
    supportNeighborhoods.computeSupportPoints(Xin, yin, Xsup);
    InternalIndexType const bs = Xsup.n_rows; // Number of receptive fields
    KDTree<InternalIndexType, InputXDecimalType> kdt =
        supportNeighborhoods.buildKDTree(Xin);
    arma::Col<arma::u8> clean(bs, arma::fill::zeros); // Clean flags
    // Prepare output
    auto out = DLPreProcessorOutput<
        InputXDecimalType, OutputXDecimalType, FDecimalType,
        LabelType, OutputIndexType
    >();
    out.xout.set_size(bs, nx);
    if(yin.n_elem > 0) out.yout.set_size(bs, R);
    out.I = std::vector<arma::Col<OutputIndexType>>(bs);
    out.Xout.push_back(std::vector<arma::Mat<OutputXDecimalType>>(bs));
    out.Fout.resize(bs);
    out.ND.push_back(std::vector<arma::Mat<OutputIndexType>>(bs));
    out.NU.push_back(std::vector<arma::Mat<OutputIndexType>>(bs));
    // Compute receptive fields
    int const chunkSize = MultithreadingUtils::correctChunkSize(
        bs, VL3DPP_OMP_CHUNK_SIZE_SMALL, nthreads
    );
    #pragma omp parallel for default(none) shared( \
            bs, nx, ny, out, supportNeighborhoods, oversampler, R, KD, \
            Xin, Fin, yin, Xsup, kdt, fps, clean, chunkSize \
        ) \
        schedule(VL3DPP_OMP_SCHEDULE, chunkSize)
    for(arma::uword i = 0 ; i  < bs ; ++i){ // Iterate over receptive fields
        fit(i, nx, Xin, Xsup, Fin, yin, kdt, fps, clean, out);
    }
    // Preserve only clean receptive fields
    cleanOutput(bs, Fin, yin, clean, out);
    // Return output
    return out;
}

template <
    typename InputXDecimalType,
    typename OutputXDecimalType,
    typename FDecimalType,
    typename InternalIndexType,
    typename OutputIndexType,
    typename LabelType
>
void
DLFPSPreProcessor<
    InputXDecimalType,
    OutputXDecimalType,
    FDecimalType,
    InternalIndexType,
    OutputIndexType,
    LabelType
>::fit(
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
){
    // Extract support point
    arma::Col<InputXDecimalType> const &x = Xsup.row(i).as_col();
    // Extract support neighborhood
    arma::Col<InternalIndexType> &Ii = out.I[i];
    supportNeighborhoods.computeNeighborhood(Xin, x, kdt, Ii);
    arma::uword mi = Ii.n_elem; // Number of points in receptive field
    // Ignore neighborhood (in Python this is called "cleaning")
    if(oversampler == nullptr && mi < R) return;
    if(oversampler != nullptr && mi < oversampler->getMinPoints()) return;
    // Extract support points and center them on x
    arma::Mat<OutputXDecimalType> Xi(mi, nx);
    for(arma::uword k = 0 ; k < mi ; ++k){
        Xi.row(k) = arma::conv_to<arma::Row<OutputXDecimalType>>::from(
            Xin.row(Ii[k]).as_col() - x
        );
    }
    // Oversample if requested and necessary
    arma::Mat<OutputXDecimalType> Yi;
    if(oversampler != nullptr && mi < R){
        Yi = (*oversampler)(Xi);
        // Compute Yi as FPS representation of oversampled Xi
        Yi = fps.sample3D(Yi);
    }
    else{
        // Compute Yi as FPS representation of Xi
        Yi = fps.sample3D(Xi);
    }
    // Now mi represents the number of points in the receptive field
    // and not in the original neighborhood anymore
    mi = Yi.n_rows;
    // Validate current receptive field
    if(mi < R){
        std::stringstream ss;
        ss  << "DLFPSPreProcessor failed to generate " << i << "-th "
            << "receptive field with " << R << " points. Only " << mi << " "
            << "points were extracted.";
        throw VL3DPPException(ss.str());
    }
    clean[i] = 1; // Flag current neighborhood as clean
    // Transform to unit sphere if requested
    if(toUnitSphere) rfield::toUnitSphere(Yi);
    // Extract neighborhood information to output
    KDTree<InternalIndexType, OutputXDecimalType> supKdt(Xi, false, true);
    arma::Mat<OutputIndexType> &N = out.ND[0][i];
    N.resize(R, KD);
    for(arma::uword k = 0 ; k < R ; ++k){
        N.row(k) = supKdt.findKnn(Yi.row(k), KD).as_row();
    }
    supKdt = KDTree<InternalIndexType, OutputXDecimalType>(Yi, false, true);
    arma::Mat<OutputIndexType> &M = out.NU[0][i];
    M.resize(Xi.n_rows, KD);
    for(arma::uword k = 0 ; k < Xi.n_rows ; ++k){
        M.row(k) = supKdt.findKnn(Xi.row(k), KD).as_row();
    }
    // Extract receptive field structure, features, and labels, to output
    out.xout.row(i) = x.as_row();
    out.Xout[0][i] = Yi;
    if(Fin.n_elem > 0){ // Handle features, if given
        arma::Mat<FDecimalType> Fi(Ii.n_rows, Fin.n_cols);
        for(arma::uword j = 0 ; j < Fin.n_cols ; ++j){
            for(arma::uword k = 0 ; k < Ii.n_rows ; ++k){
                Fi.at(k, j) = Fin.at(Ii[k], j);
            }
        }
        out.Fout[i] = rfield::reduceMean(N, Fi);
    }
    if(yin.n_elem > 0){ // Handle labels if given
        arma::Col<LabelType> yi(Ii.n_rows);
        for(arma::uword k = 0 ; k < Ii.n_rows ; ++k){
            yi[k] = yin[Ii[k]];
        }
        out.yout.row(i) = rfield::reduceLabelMode(N, yi, ny).as_row();
    }
}


// ***  UTIL METHODS  *** //
// ********************** //
template<
    typename InputXDecimalType,
    typename OutputXDecimalType,
    typename FDecimalType,
    typename InternalIndexType,
    typename OutputIndexType,
    typename LabelType
>
void
DLFPSPreProcessor<
    InputXDecimalType,
    OutputXDecimalType,
    FDecimalType,
    InternalIndexType,
    OutputIndexType,
    LabelType
>::cleanOutput(
    InternalIndexType const bs,
    arma::Mat<FDecimalType> const &Fin,
    arma::Col<LabelType> const &yin,
    arma::Col<arma::u8> const &clean,
    DLPreProcessorOutput<
        InputXDecimalType, OutputXDecimalType, FDecimalType, LabelType,
        OutputIndexType
    > &out
){
    // Find indices of clean receptive fields
    arma::Col<arma::uword> const cleanIndices = arma::find(clean);
    // Prepare clean outputs for X, F, I
    std::vector<arma::Col<OutputIndexType>> Iout;
    Iout.reserve(cleanIndices.n_elem);
    std::vector<arma::Mat<OutputXDecimalType>> Xout;
    Xout.reserve(cleanIndices.n_elem);
    std::vector<arma::Mat<FDecimalType>> Fout;
    if(Fin.n_elem > 0) Fout.reserve(cleanIndices.n_elem);
    std::vector<arma::Mat<OutputIndexType>> Nbuffer; // Used to store I here
    Nbuffer.reserve(cleanIndices.n_elem);
    // Update output in place for X, F, I, N
    for(size_t i = 0 ; i < bs ; ++i){
        if(!clean[i]) continue;
        Iout.push_back(out.I[i]);
        Xout.push_back(out.Xout[0][i]);
        if(Fin.n_elem > 0) Fout.push_back(out.Fout[i]);
        Nbuffer.push_back(out.ND[0][i]);
    }
    out.I = std::move(Iout);
    out.Xout[0] = std::move(Xout);
    if(Fin.n_elem > 0) out.Fout = std::move(Fout);
    out.ND[0] = Nbuffer;
    Nbuffer.clear();
    // Update output in place for M
    for(size_t i = 0 ; i < bs ; ++i){
        if(!clean[i]) continue;
        Nbuffer.push_back(out.NU[0][i]);
    }
    out.NU[0] = Nbuffer;
    // Update output in place for x and y
    out.xout = out.xout.rows(cleanIndices);
    if(yin.n_elem > 0) out.yout = out.yout.rows(cleanIndices);
}

