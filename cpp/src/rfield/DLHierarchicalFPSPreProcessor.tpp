#include <rfield/DLHierarchicalFPSPreProcessor.hpp>

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
DLHierarchicalFPSPreProcessor<
    InputXDecimalType,
    OutputXDecimalType,
    FDecimalType,
    InternalIndexType,
    OutputIndexType,
    LabelType
>::DLHierarchicalFPSPreProcessor(
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
) :
    supportNeighborhoods(supportNeighborhoods),
    ny(ny),
    toUnitSphere(toUnitSphere),
    R(R),
    KD(KD),
    KU(KU),
    KN(KN),
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
DLHierarchicalFPSPreProcessor<
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
    InternalIndexType const maxDepth = R.n_elem; // Max depth of the hierarchy
    InternalIndexType const nx = Xin.n_cols; // Structure space dimensionality
    std::vector<FurthestPointSubsampler<OutputXDecimalType>> hfps;
    for(InternalIndexType d = 0 ; d < maxDepth ; ++d){
        hfps.push_back(FurthestPointSubsampler<OutputXDecimalType>(R[d], 1));
    }
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
    if(yin.n_elem > 0) out.yout.set_size(bs, R[0]);
    out.I = std::vector<arma::Col<OutputIndexType>>(bs);
    for(InternalIndexType d = 0 ; d < maxDepth ; ++d){
        out.Xout.push_back(std::vector<arma::Mat<OutputXDecimalType>>(bs));
        out.ND.push_back(std::vector<arma::Mat<OutputIndexType>>(bs));
        out.NU.push_back(std::vector<arma::Mat<OutputIndexType>>(bs));
        out.N.push_back(std::vector<arma::Mat<OutputIndexType>>(bs));
    }
    out.Fout.resize(bs);
    // Compute receptive fields
    int const chunkSize = MultithreadingUtils::correctChunkSize(
        bs, VL3DPP_OMP_CHUNK_SIZE_SMALL, nthreads
    );
    #pragma omp parallel for default(none) shared( \
            bs, nx, ny, out, supportNeighborhoods, oversampler, maxDepth, R, \
            KD, KU, KN, Xin, Fin, yin, Xsup, kdt, hfps, clean, chunkSize \
        ) \
        schedule(VL3DPP_OMP_SCHEDULE, chunkSize)
    for(arma::uword i = 0 ; i  < bs ; ++i){ // Iterate over receptive fields
        fit(i, maxDepth, nx, Xin, Xsup, Fin, yin, kdt, hfps, clean, out);
    }
    // Preserve only clean receptive fields
    cleanOutput(bs, maxDepth, Fin, yin, clean, out);
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
DLHierarchicalFPSPreProcessor<
    InputXDecimalType,
    OutputXDecimalType,
    FDecimalType,
    InternalIndexType,
    OutputIndexType,
    LabelType
>::fit(
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
){
    // Extract support point
    arma::Col<InputXDecimalType> const &x = Xsup.row(i).as_col();
    // Extract support neighborhood
    arma::Col<InternalIndexType> &Ii = out.I[i];
    supportNeighborhoods.computeNeighborhood(Xin, x, kdt, Ii);
    arma::uword mi = Ii.n_elem; // Number of points in receptive field
    // Ignore neighborhood (in Python this is called "cleaning")
    if(oversampler == nullptr && mi < R[0]) return;
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
    if(oversampler != nullptr && mi < R[0]){
        Yi = (*oversampler)(Xi);
        // Compute Yi as FPS representation of oversampled Xi
        Yi = hfps[0].sample3D(Yi);
    }
    else{
        // Compute Yi as FPS representation of Xi
        Yi = hfps[0].sample3D(Xi);
    }
    // Now mi represents the number of points in the receptive field
    // and not in the original neighborhood anymore
    mi = Yi.n_rows;
    // Validate current receptive field
    if(mi < R[0]){
        std::stringstream ss;
        ss  << "DLHierarchicalFPSPreProcessor failed to generate " << i
            << "-th receptive field with " << R[0] << " points. Only " << mi
            << " points were extracted.";
        throw VL3DPPException(ss.str());
    }
    clean[i] = 1; // Flag current neighborhood as clean
    // Transform to unit sphere if requested
    if(toUnitSphere) rfield::toUnitSphere(Yi);
    // Extract receptive field center to output
    out.xout.row(i) = x.as_row();
    // Extract neighborhood information for each depth to output
    for(InternalIndexType d = 0 ; d < maxDepth ; ++d){
        // Receptive field's structure space
        out.Xout[d][i] = (d==0) ? Yi : hfps[d].sample3D(out.Xout[d-1][i]);
        // Transform to unit sphere if requested
        if(toUnitSphere) rfield::toUnitSphere(out.Xout[d][i]);
        arma::Mat<OutputXDecimalType> const &Yid = out.Xout[d][i];
        // Downsampling neighborhoods
        KDTree<InternalIndexType, OutputXDecimalType> supKdt(
            (d==0) ? Xi : out.Xout[d-1][i],
            false,
            true
        );
        arma::Mat<OutputIndexType> &NDd = out.ND[d][i];
        NDd.resize(R[d], KD[d]);
        for(arma::uword k = 0 ; k < R[d] ; ++k){
            NDd.row(k) = supKdt.findKnn(Yid.row(k), KD[d]).as_row();
        }
        // Upsampling neighborhoods
        supKdt = KDTree<InternalIndexType, OutputXDecimalType>(
            Yid, false, true
        );
        arma::Mat<OutputIndexType> &NUd = out.NU[d][i];
        arma::uword const dim = (d==0) ? Xi.n_rows : out.Xout[d-1][i].n_rows;
        NUd.resize(dim, KU[d]);
        if(d==0){
            for(arma::uword k = 0 ; k < dim ; ++k){
                NUd.row(k) = supKdt.findKnn(Xi.row(k), KU[d]).as_row();
            }
        }
        else{
            arma::Mat<OutputXDecimalType> const &Ydprevi = out.Xout[d-1][i];
            for(arma::uword k = 0 ; k < dim ; ++k){
                NUd.row(k) = supKdt.findKnn(Ydprevi.row(k), KU[d]).as_row();
            }

        }
        // Neighborhoods at a given depth
        arma::Mat<OutputIndexType> &Nd = out.N[d][i];
        Nd.resize(R[d], KN[d]);
        for(arma::uword k = 0 ; k < Yid.n_rows ; ++k){
            Nd.row(k) = supKdt.findKnn(Yid.row(k), KN[d]).as_row();
        }
    }
    // Extract features and labels to output
    arma::Mat<OutputIndexType> &ND0 = out.ND[0][i];
    if(Fin.n_elem > 0){ // Handle features, if given
        arma::Mat<FDecimalType> Fi(Ii.n_rows, Fin.n_cols);
        for(arma::uword j = 0 ; j < Fin.n_cols ; ++j){
            for(arma::uword k = 0 ; k < Ii.n_rows ; ++k){
                Fi.at(k, j) = Fin.at(Ii[k], j);
            }
        }
        out.Fout[i] = rfield::reduceMean(ND0, Fi);
    }
    if(yin.n_elem > 0){ // Handle labels if given
        arma::Col<LabelType> yi(Ii.n_rows);
        for(arma::uword k = 0 ; k < Ii.n_rows ; ++k){
            yi[k] = yin[Ii[k]];
        }
        out.yout.row(i) = rfield::reduceLabelMode(ND0, yi, ny).as_row();
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
DLHierarchicalFPSPreProcessor<
    InputXDecimalType,
    OutputXDecimalType,
    FDecimalType,
    InternalIndexType,
    OutputIndexType,
    LabelType
>::cleanOutput(
    InternalIndexType const bs,
    InternalIndexType const maxDepth,
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
    // Prepare depth-independent clean outputs
    std::vector<arma::Col<OutputIndexType>> Iout;
    Iout.reserve(cleanIndices.n_elem);
    std::vector<arma::Mat<FDecimalType>> Fout;
    if(Fin.n_elem > 0) Fout.reserve(cleanIndices.n_elem);
    for(size_t i = 0 ; i < bs ; ++i){
        if(!clean[i]) continue;
        Iout.push_back(out.I[i]);
        if(Fin.n_elem > 0) Fout.push_back(out.Fout[i]);
    }
    // Update depth-independent output in place, preserving only clean rfields
    out.I = std::move(Iout);
    if(Fin.n_elem > 0) out.Fout = std::move(Fout);
    out.xout = out.xout.rows(cleanIndices);
    if(yin.n_elem > 0) out.yout = out.yout.rows(cleanIndices);
    // Handle depth-dependent clean outputs
    std::vector<arma::Mat<OutputXDecimalType>> Xbuffer;
    Xbuffer.reserve(cleanIndices.n_elem);
    std::vector<arma::Mat<OutputIndexType>> Nbuffer;
    Nbuffer.reserve(cleanIndices.n_elem);
    for(InternalIndexType d = 0 ; d < maxDepth ; ++d){
        // Prepare depth-dependent clean outputs for X and ND
        for(size_t i = 0 ; i < bs ; ++i){
            if(!clean[i]) continue;
            Xbuffer.push_back(out.Xout[d][i]);
            Nbuffer.push_back(out.ND[d][i]);
        }
        // Update depth-dependent X and ND in place, preserve only clean rfields
        out.Xout[d] = Xbuffer;
        Xbuffer.clear();
        out.ND[d] = Nbuffer;
        Nbuffer.clear();
        // Prepare depth-dependent clean outputs for NU
        for(size_t i = 0 ; i < bs ; ++i){
            if(!clean[i]) continue;
            Nbuffer.push_back(out.NU[d][i]);
        }
        // Update depth-dependent NU in place, preserve only clean rfields
        out.NU[d] = Nbuffer;
        Nbuffer.clear();
        // Prepare depth-dependent clean outputs for N
        for(size_t i = 0 ; i < bs ; ++i){
            if(!clean[i]) continue;
            Nbuffer.push_back(out.N[d][i]);
        }
        // Update depth-dependent NU in place, preserve only clean rfields
        out.N[d] = Nbuffer;
        Nbuffer.clear();
    }
}
