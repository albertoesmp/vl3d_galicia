#ifndef _RECEPTIVE_FIELD_MODULE_
#define _RECEPTIVE_FIELD_MODULE_

/**
 * @author Alberto M. Esmoris Pena
 *
 * Provides functions wrapping VL3DPP to be easily called from the VL3D
 * python software.
 * More concretely, the functions here wrap receptive field components.
 */

// ***   INCLUDES   *** //
// ******************** //
#include <module/ReceptiveFieldModuleCommon.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <carma>
#include <armadillo>

#include <string>
#include <vector>

#include <rfield/ReceptiveFieldCommon.hpp>
#include <rfield/DLFPSPreProcessor.hpp>
#include <rfield/DLHierarchicalFPSPreProcessor.hpp>
#include <rfield/DLHierarchicalSGPreProcessor.hpp>
#include <alg/Oversampler.hpp>
#include <rfield/DLPreProcessorOutput.hpp>
#include <rfield/DLFPSPostProcessor.hpp>
#include <rfield/DLSGPostProcessor.hpp>
#include <adt/grid/HierarchicalSparseGrid.hpp>
#include <util/VL3DPPMacros.hpp>
#include <util/MultithreadingUtils.hpp>

#include <thread>

namespace py = pybind11;
using namespace vl3dpp::rfield;
using namespace vl3dpp::alg;

namespace vl3dpp::pymods {




// ***   MODULE CALLS   *** //
// ************************ //
template <typename FDecimalType, typename IndexType>
py::array rf_propagate_mean(
    py::array const &M,
    py::array const &F
){
    arma::Mat<FDecimalType> Fprop = rfield::propagateMean(
        carma::arr_to_mat_view<IndexType>(M),
        carma::arr_to_mat_view<FDecimalType>(F)
    );
    return carma::mat_to_arr(Fprop, false);
}

template <typename FDecimalType, typename IndexType>
py::array rf_propagate_closest(py::array const &M, py::array const &F){
    arma::Mat<FDecimalType> Fprop = rfield::propagateClosest(
        carma::arr_to_mat_view<IndexType>(M),
        carma::arr_to_mat_view<FDecimalType>(F)
    );
    return carma::mat_to_arr(Fprop, false);
}

template <typename FDecimalType, typename IndexType>
py::array rf_reduce_mean(
    py::array const &N, py::array const &F
){
    arma::Mat<FDecimalType> Fred = rfield::reduceMean(
        carma::arr_to_mat_view<IndexType>(N),
        carma::arr_to_mat_view<FDecimalType>(F)
    );
    return carma::mat_to_arr(Fred, false);
}

template <
    typename LabelType,
    typename OriginalIndexType,
    typename InternalIndexType
>
py::array rf_reduce_label_mode(
    py::list const &I,
    py::list const &N,
    py::array const &y,
    LabelType const ny,
    int const nthreads
){
    // Extract arguments
    size_t const numReceptiveFields = py::len(I);
    size_t const numPointsPerRF = carma::arr_to_mat_view<InternalIndexType>(
        N[0].cast<py::array>()
    ).n_rows;
    arma::Col<LabelType> const &_y = carma::arr_to_col_view<LabelType>(y);
    arma::Mat<LabelType> out(numReceptiveFields, numPointsPerRF);
    std::vector<arma::Col<arma::uword>> _I(numReceptiveFields);
    std::vector<arma::Mat<InternalIndexType>> _N(numReceptiveFields);
    for(size_t i = 0 ; i < numReceptiveFields ; ++i){
        _I[i] = arma::conv_to<arma::Col<arma::uword>>::from(
            carma::arr_to_col_view<OriginalIndexType>(I[i].cast<py::array>())
        );
        _N[i] = carma::arr_to_mat_view<InternalIndexType>(
            N[i].cast<py::array>()
        );
    }
    // Handle as many threads as available if requested
    unsigned int _nthreads = nthreads;
    if(nthreads==-1) _nthreads = std::thread::hardware_concurrency();
    // Compute reduction in parallel
    omp_set_num_threads(_nthreads); // Use nthreads parallel threads at most
    #pragma omp parallel for default(none) shared( \
            numReceptiveFields, _y, _N, out, ny, _I \
        ) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED_SMALL)
    for(size_t i = 0 ; i < numReceptiveFields ; ++i){
        arma::Col<arma::uword> const &Ii = _I[i];
        arma::Col<LabelType> const &yIi = _y.rows(Ii);
        arma::Mat<InternalIndexType> const &Ni = _N[i];
        out.row(i) = rfield::reduceLabelMode(Ni, yIi, ny).as_row();
    }
    // Return output to Python
    return carma::mat_to_arr(out, false);
}

template <
    typename XDecimalType,
    typename LabelType,
    typename IndexType
>
py::list rf_sparse_reduce_label_mode(
    py::array const &X,
    py::array const &y,
    py::array const &size,
    py::list const &A,
    py::list const &n,
    py::list const &hk,
    py::list const &hv,
    int const nthreads
){
    // Extract arguments
    size_t const numReceptiveFields = py::len(A);
    arma::Mat<XDecimalType> _X = carma::arr_to_mat_view<XDecimalType>(X);
    arma::Col<LabelType> const &_y = carma::arr_to_col_view<LabelType>(y);
    arma::Col<XDecimalType> const &_size = carma::arr_to_col_view<XDecimalType>(
        size
    );
    std::vector<arma::Row<XDecimalType>> _A;
    std::vector<arma::Col<IndexType>> _n;
    std::vector<arma::Col<IndexType>> _hk;
    std::vector<arma::Col<IndexType>> _hv;
    for(size_t i = 0 ; i < numReceptiveFields ; ++i){
        _A.push_back(
            carma::arr_to_row_view<XDecimalType>(A[i].cast<py::array>())
        );
        _n.push_back(
            carma::arr_to_col_view<IndexType>(n[i].cast<py::array>())
        );
        _hk.push_back(
            carma::arr_to_col_view<IndexType>(hk[i].cast<py::array>())
        );
        _hv.push_back(carma::arr_to_col_view<IndexType>(
            hv[i].cast<py::array>())
        );
    }
    // Prepare output
    std::vector<arma::Col<LabelType>> out(numReceptiveFields);
    // Prepare computation
    // Handle as many threads as available, if requested
    int _nthreads = nthreads;
    if(nthreads==-1) _nthreads = std::thread::hardware_concurrency();
    // Compute reduction in parallel
    omp_set_num_threads(_nthreads);  // Use nthreads parallel threads at most
    int const chunkSize = MultithreadingUtils::correctChunkSize(
        numReceptiveFields, VL3DPP_OMP_CHUNK_SIZE_SMALL, _nthreads
    );
    #pragma omp parallel for default(none) shared( \
        numReceptiveFields, _X, _y, _A, _size, _n, _hk, _hv, chunkSize, out \
    ) schedule(VL3DPP_OMP_SCHEDULE, chunkSize)
    for(size_t i = 0 ; i < numReceptiveFields ; ++i){
        XDecimalType const sizei = _size[i];
        arma::Row<XDecimalType> const &Ai = _A[i];
        arma::Col<IndexType> const &ni = _n[i];
        arma::Col<IndexType> const &hki = _hk[i];
        arma::Col<IndexType> const &hvi = _hv[i];
        SparseGrid sg(sizei, Ai, ni, hki, hvi);
        arma::Row<XDecimalType> const Bi = Ai + arma::conv_to<
            arma::Row<XDecimalType>
        >::from(ni) * sizei;
        std::vector<arma::uword> pibb(0); // Points in bounding box
        pibb.reserve(_X.n_rows);
        for(arma::uword p = 0 ; p < _X.n_rows ; ++p){
            arma::Row<XDecimalType> const & xp = _X.row(p);
            bool inside = true;
            for(arma::uword j = 0 ; j < xp.n_cols ; ++j){
                inside &= xp[j] >= Ai[j] && xp[j] <= Bi[j];
            }
            if(inside) pibb.push_back(p);
        }
        arma::uvec const pointsInBoundingBox =
            arma::conv_to<arma::uvec>::from(pibb);
        arma::Mat<XDecimalType> const Xi = _X.rows(pointsInBoundingBox);
        arma::Col<LabelType> const yi = _y.rows(pointsInBoundingBox);
        out[i] = sg.template encodeVector<LabelType>(Xi, yi, "mode");
    }
    // Transform output to python list
    py::list pyout;
    for(arma::Col<LabelType> const & outi : out){
        pyout.append(carma::col_to_arr<LabelType>(outi));
    };
    // Return encoded labels
    return pyout;
}


template <
    typename FDecimalType,
    typename EncodingIndexType,
    typename DecodingIndexType
>
py::array rf_dl_fps_postproc_mean(
    std::string const &reductionType,
    arma::uword const m,
    py::array const &zBatch,
    py::list const &MBatch,
    py::list const &I,
    FDecimalType const minClipValue,
    int const nthreads
){
    // Convert
    std::vector<arma::Mat<EncodingIndexType>> _MBatch;
    _MBatch.reserve(py::len(MBatch));
    std::vector<arma::Col<DecodingIndexType>> _I;
    _I.reserve(py::len(I));
    for(auto it = MBatch.begin() ; it != MBatch.end() ; ++it){
        auto arr = it->cast<py::array>();
        _MBatch.push_back(carma::arr_to_mat_view<EncodingIndexType>(arr));
    }
    for(auto it = I.begin() ; it != I.end() ; ++it){
        auto arr = it->cast<py::array>();
        _I.push_back(carma::arr_to_col_view<DecodingIndexType>(arr));
    }
    // Instantiate post-processor
    rfield::DLFPSPostProcessor<
        FDecimalType, EncodingIndexType, DecodingIndexType
    > dlFPSPostProc(
        reductionType,
        minClipValue,
        nthreads
    );
    // Compute post-processed probabilities
    arma::Mat<FDecimalType> out = dlFPSPostProc(
        m,
        _MBatch,
        _I,
        carma::arr_to_cube_view<FDecimalType>(zBatch)
    );
    // Return post-processed probabilities
    return carma::mat_to_arr(out, false);
}

template <
    typename InputXDecimalType,
    typename OutputXDecimalType,
    typename FDecimalType,
    typename InternalIndexType,
    typename OutputIndexType,
    typename LabelType
>
py::list rf_dl_fps_preproc(
    py::array const &Xin,
    py::array const &Fin,
    py::array const &yin,
    LabelType const ny,
    bool const toUnitSphere,
    InternalIndexType const R,
    InternalIndexType const KD,
    short const fast,
    py::list supportArgs,
    py::list oversamplingArgs,
    int nthreads
){
    // Convert
    arma::Mat<InputXDecimalType> _Xin = \
        carma::arr_to_mat_view<InputXDecimalType>(Xin);
    arma::Mat<FDecimalType> _Fin = carma::arr_to_mat_view<FDecimalType>(Fin);
    arma::Col<LabelType> _yin = carma::arr_to_col_view<LabelType>(yin);
    // Instantiate support neighborhoods
    SupportNeighborhoods<InputXDecimalType, LabelType, InternalIndexType> sn =
        instantiateSupportNeighborhoods<
            InputXDecimalType, LabelType, InternalIndexType
        >(supportArgs, nthreads);
    // Instantiate oversampler
    Oversampler<OutputXDecimalType, OutputIndexType> *oversampler = nullptr;
    if(py::len(oversamplingArgs) > 0){
        oversampler = new Oversampler<OutputXDecimalType, OutputIndexType>(
            oversamplingArgs[0].cast<OutputIndexType>(), // Min points
            oversamplingArgs[1].cast<OutputIndexType>(), // Target points
            oversamplingArgs[2].cast<std::string>(), // Strategy
            oversamplingArgs[3].cast<OutputIndexType>(), // K
            oversamplingArgs[4].cast<OutputXDecimalType>() // Radius
        );
    }
    // Instantiate pre-processor
    rfield::DLFPSPreProcessor<
        InputXDecimalType,
        OutputXDecimalType,
        FDecimalType,
        InternalIndexType,
        OutputIndexType,
        LabelType
    > dlFPSPreProc(sn, ny, toUnitSphere, R, KD, fast, oversampler, nthreads);
    // Compute pre-processed receptive fields
    DLPreProcessorOutput<
        InputXDecimalType,
        OutputXDecimalType,
        FDecimalType,
        LabelType,
        OutputIndexType
    > outpp = dlFPSPreProc(_Xin, _Fin, _yin);
    // Wrap output in Python list
    py::list outpy;
    outpy.append(carma::mat_to_arr<InputXDecimalType>(std::move(outpp.xout))); // x
    py::list Xout;
    for(arma::Mat<OutputXDecimalType> &Xouti : outpp.Xout[0]){
        Xout.append(carma::mat_to_arr<OutputXDecimalType>(std::move(Xouti)));
    }
    outpy.append(std::move(Xout)); // X
    py::list Fout;
    for(arma::Mat<FDecimalType> &Fouti : outpp.Fout){
        Fout.append(carma::mat_to_arr<FDecimalType>(std::move(Fouti)));
    }
    outpy.append(std::move(Fout)); // F
    outpy.append(carma::mat_to_arr<LabelType>(outpp.yout)); // y
    py::list Nout;
    for(arma::Mat<OutputIndexType> &NDi : outpp.ND[0]){
        Nout.append(carma::mat_to_arr<OutputIndexType>(std::move(NDi)));
    }
    outpy.append(std::move(Nout)); // N
    py::list Mout;
    for(arma::Mat<OutputIndexType> &MDi : outpp.NU[0]){
        Mout.append(carma::mat_to_arr<OutputIndexType>(std::move(MDi)));
    }
    outpy.append(std::move(Mout)); // M
    py::list Ipy;
    for(arma::Col<OutputIndexType> &Ii : outpp.I){
        Ipy.append(carma::col_to_arr<OutputIndexType>(std::move(Ii)));
    }
    outpy.append(std::move(Ipy)); // I
    // Release memory
    delete oversampler;
    // Return output as a Python list
    return outpy;
}

template <
    typename InputXDecimalType,
    typename OutputXDecimalType,
    typename FDecimalType,
    typename InternalIndexType,
    typename OutputIndexType,
    typename LabelType
>
py::list rf_dl_hfps_preproc(
    py::array const &Xin,
    py::array const &Fin,
    py::array const &yin,
    LabelType const ny,
    bool const toUnitSphere,
    py::array const &R,
    py::array const &KD,
    py::array const &KU,
    py::array const &KN,
    py::list const fast,
    py::list const supportArgs,
    py::list const oversamplingArgs,
    int const nthreads
){
    // Convert
    arma::Mat<InputXDecimalType> _Xin = \
        carma::arr_to_mat_view<InputXDecimalType>(Xin);
    arma::Mat<FDecimalType> _Fin = carma::arr_to_mat_view<FDecimalType>(Fin);
    arma::Col<LabelType> _yin = carma::arr_to_col_view<LabelType>(yin);
    std::vector<short> _fast;
    for(auto it = fast.begin() ; it != fast.end() ; ++it){
        _fast.push_back(it->cast<short>());
    }
    // Instantiate support neighborhoods
    SupportNeighborhoods<InputXDecimalType, LabelType, InternalIndexType> sn =
        instantiateSupportNeighborhoods<
            InputXDecimalType, LabelType, InternalIndexType
        >(supportArgs, nthreads);
    // Instantiate oversampler
    Oversampler<OutputXDecimalType, OutputIndexType> *oversampler = nullptr;
    if(py::len(oversamplingArgs) > 0){
        oversampler = new Oversampler<OutputXDecimalType, OutputIndexType>(
            oversamplingArgs[0].cast<OutputIndexType>(), // Min points
            oversamplingArgs[1].cast<OutputIndexType>(), // Target points
            oversamplingArgs[2].cast<std::string>(), // Strategy
            oversamplingArgs[3].cast<OutputIndexType>(), // K
            oversamplingArgs[4].cast<OutputXDecimalType>() // Radius
        );
    }
    // Instantiate pre-processor
    rfield::DLHierarchicalFPSPreProcessor<
        InputXDecimalType,
        OutputXDecimalType,
        FDecimalType,
        InternalIndexType,
        OutputIndexType,
        LabelType
    > dlHierarchicalFPSPreProc(
        sn,
        ny,
        toUnitSphere,
        carma::arr_to_col_view<InternalIndexType>(R),
        carma::arr_to_col_view<InternalIndexType>(KD),
        carma::arr_to_col_view<InternalIndexType>(KU),
        carma::arr_to_col_view<InternalIndexType>(KN),
        _fast,
        oversampler,
        nthreads
    );
    // Compute pre-processed receptive fields
    DLPreProcessorOutput<
        InputXDecimalType,
        OutputXDecimalType,
        FDecimalType,
        LabelType,
        OutputIndexType
    > outpp = dlHierarchicalFPSPreProc(_Xin, _Fin, _yin);
    // Wrap depth-independent output in Python list
    py::list outpy;
    outpy.append(carma::mat_to_arr<InputXDecimalType>(std::move(outpp.xout))); // x
    outpp.xout.reset();
    py::list Fout; // F
    for(arma::Mat<FDecimalType> &Fouti : outpp.Fout){
        Fout.append(carma::mat_to_arr<FDecimalType>(std::move(Fouti)));
    }
    outpy.append(std::move(Fout));
    outpp.Fout = std::vector<arma::Mat<FDecimalType>>(0);
    outpy.append(carma::mat_to_arr<LabelType>(std::move(outpp.yout))); // y
    outpp.yout.reset();
    py::list Ipy; // I
    for(arma::Col<OutputIndexType> &Ii : outpp.I){
        Ipy.append(carma::col_to_arr<OutputIndexType>(std::move(Ii)));
    }
    outpy.append(std::move(Ipy));
    outpp.I = std::vector<arma::Col<OutputIndexType>>(0);
    // Wrap depth-dependent output in Python list
    InternalIndexType const maxDepth = _fast.size();
    py::list Xout;
    py::list NDout, NUout, Nout;
    for(InternalIndexType d = 0 ; d < maxDepth ; ++d){
        py::list Xoutd; // X (Y)
        for(arma::Mat<OutputXDecimalType> &Xouti : outpp.Xout[d]){
            Xoutd.append(
                carma::mat_to_arr<OutputXDecimalType>(std::move(Xouti))
            );
        }
        outpp.Xout[d] = std::vector<arma::Mat<OutputXDecimalType>>(0);
        Xout.append(std::move(Xoutd));
        py::list NDdout; // ND
        for(arma::Mat<OutputIndexType> &NDi : outpp.ND[d]){
            NDdout.append(carma::mat_to_arr<OutputIndexType>(std::move(NDi)));
        }
        outpp.ND[d] = std::vector<arma::Mat<OutputIndexType>>(0);
        NDout.append(std::move(NDdout));
        py::list NUdout; // NU
        for(arma::Mat<OutputIndexType> &NUi : outpp.NU[d]){
            NUdout.append(carma::mat_to_arr<OutputIndexType>(std::move(NUi)));
        }
        outpp.NU[d] = std::vector<arma::Mat<OutputIndexType>>(0);
        NUout.append(std::move(NUdout));
        py::list Ndout; // N
        for(arma::Mat<OutputIndexType> &Ni : outpp.N[d]){
            Ndout.append(carma::mat_to_arr<OutputIndexType>(std::move(Ni)));
        }
        outpp.N[d] = std::vector<arma::Mat<OutputIndexType>>(0);
        Nout.append(std::move(Ndout));
    }
    outpp = DLPreProcessorOutput< // No longer needed, swap by empty instance
        InputXDecimalType,
        OutputXDecimalType,
        FDecimalType,
        LabelType,
        OutputIndexType
    >();
    outpy.append(std::move(Xout)); // X
    outpy.append(std::move(NDout)); // ND
    outpy.append(std::move(NUout)); // NU
    outpy.append(std::move(Nout)); // N
    // Release memory
    delete oversampler;
    // Return output as a Python list
    return outpy;
}

template <
    typename XDecimalType,
    typename IndexType
>
py::list rf_dl_sg_fit(
    py::array const &X,
    XDecimalType const size,
    py::array const &w,
    py::array const &wD,
    py::array const &wU,
    py::array const &sD,
    py::array const &sU,
    int const nthreads
){
    // Convert
    arma::Col<IndexType> const _w = carma::arr_to_col_view<IndexType>(w);
    arma::Col<IndexType> const _wD = carma::arr_to_col_view<IndexType>(wD);
    arma::Col<IndexType> const _wU = carma::arr_to_col_view<IndexType>(wU);
    arma::Col<IndexType> const _sD = carma::arr_to_col_view<IndexType>(sD);
    arma::Col<IndexType> const _sU = carma::arr_to_col_view<IndexType>(sU);
    // Instantiate HierarchicalSparseGrid
    adt::grid::HierarchicalSparseGrid<XDecimalType, IndexType> hsg(
        size, _w, _wD, _wU, _sD, _sU, nthreads
    );
    // Fit HierarchicalSparseGrid
    arma::Mat<XDecimalType> const _X = carma::arr_to_mat_view<XDecimalType>(X);
    arma::uword const nx = _X.n_cols;
    hsg.fit(_X);
    // Build output
    py::list outpy;
    py::list h;
    py::list hU;
    py::list hD;
    py::array A = carma::row_to_arr(hsg.getMinVertex());
    size_t const maxDepth = hsg.getMaxDepth();
    arma::Mat<IndexType> n(nx, maxDepth);
    for(size_t t = 0 ; t < maxDepth ; ++t){
        py::dict ht = py::cast(hsg.getMap(t));
        h.append(ht);
        n.col(t) = hsg.getNumAxisPartitions(t);
        if(t < (maxDepth-1)){
            hU.append(carma::col_to_arr<IndexType>(hsg.getDownsamplingMap(t)));
            hD.append(carma::col_to_arr<IndexType>(hsg.getUpsamplingMap(t)));
        }
    }
    outpy.append(h);
    outpy.append(hU);
    outpy.append(hD);
    outpy.append(n);
    outpy.append(A);
    return outpy;
}

template<
    typename XDecimalType,
    typename FDecimalType,
    typename IndexType,
    typename LabelType
>
py::list rf_dl_hsg_preproc(
    py::array const &Xin,
    py::array const &Fin,
    py::array const &yin,
    LabelType const ny,
    XDecimalType const size,
    py::array const &w,
    py::array const &wD,
    py::array const &wU,
    py::array const &sD,
    py::array const &sU,
    py::list const &supportArgs,
    int const nthreads
){
    // Convert
    arma::Mat<XDecimalType> const _Xin = carma::arr_to_mat_view<XDecimalType>(
        Xin
    );
    arma::uword const nx = _Xin.n_cols;
    arma::Mat<FDecimalType> const _Fin = carma::arr_to_mat_view<FDecimalType>(
        Fin
    );
    arma::Col<LabelType> const _yin = carma::arr_to_col_view<LabelType>(yin);
    // Instantiate support neighborhoods
    SupportNeighborhoods<XDecimalType, LabelType, IndexType> sn =
        instantiateSupportNeighborhoods<XDecimalType, LabelType, IndexType>(
            supportArgs, nthreads
        );
    // Instantiate pre-processor
    rfield::DLHierarchicalSGPreProcessor<
        XDecimalType, FDecimalType, IndexType, LabelType
    > dlHierarchicalSGPreProc(
        sn,
        ny,
        size,
        carma::arr_to_col_view<IndexType>(w),
        carma::arr_to_col_view<IndexType>(wD),
        carma::arr_to_col_view<IndexType>(wU),
        carma::arr_to_col_view<IndexType>(sD),
        carma::arr_to_col_view<IndexType>(sU),
        nthreads
    );
    // Compute pre-processed receptive fields
    DLSparsePreProcessorOutput<
        XDecimalType,
        FDecimalType,
        LabelType,
        IndexType
    > outpp = dlHierarchicalSGPreProc(_Xin, _Fin, _yin);
    // Wrap output in Python list
    py::list outpy;
    py::list Fout; // F
    for(arma::Mat<FDecimalType> &Fouti : outpp.Fout){
        if(Fouti.n_rows > 0 && Fouti.n_cols > 0){
            Fout.append(carma::mat_to_arr<FDecimalType>(std::move(Fouti)));
        }
    }
    outpy.append(std::move(Fout));
    outpp.Fout = std::vector<arma::Mat<FDecimalType>>(0);
    py::list yout; // y
    for(arma::Col<LabelType> &youti : outpp.yout){
        if(youti.n_rows > 0){
            yout.append(carma::col_to_arr<LabelType>(std::move(youti)));
        }
    }
    outpy.append(std::move(yout));
    outpp.yout = std::vector<arma::Col<LabelType>>(0);
    py::list hout; // h
    py::list hDout; // hD
    py::list hUout; // hU
    py::list n; // n
    py::list A; // A
    arma::uword maxDepth = outpp.hsg[0].getMaxDepth();
    for(HierarchicalSparseGrid<XDecimalType, IndexType> &hsgk : outpp.hsg){
        py::list houtk; // hk (h for k-th receptive field)
        py::list hDoutk; // hDk (hD for k-th receptive field)
        py::list hUoutk; // hUk (hU for k-th receptive field)
        arma::Mat<IndexType> nk( // nk (n for k-th receptive field)
            nx, hsgk.getMaxDepth()
        );
        for(arma::uword t = 0 ; t < (maxDepth-1) ; ++t){
            hDoutk.append(carma::col_to_arr<IndexType>(
                hsgk.getDownsamplingMap(t)
            ));
            hUoutk.append(carma::col_to_arr<IndexType>(
                hsgk.getUpsamplingMap(t)
            ));
        }
        for(arma::uword t = 0 ; t < maxDepth ; ++t){
            for(arma::uword j = 0 ; j < nx ; ++j){
                nk.at(j, t) = hsgk.getNumAxisPartitions(t, j);
            }
            py::dict houtkt = py::cast( // hkt (ht for k-th receptive field)
                hsgk.getMap(t)
            );
            houtk.append(houtkt);
        }
        // Append k-th receptive field
        hout.append(houtk);
        hDout.append(hDoutk);
        hUout.append(hUoutk);
        n.append(carma::mat_to_arr<IndexType>(nk));
        A.append(carma::row_to_arr<XDecimalType>(hsgk.getMinVertex()));
        // TODO Rethink : Release hsgk memory ?
    }
    outpy.append(hout);
    outpy.append(hDout);
    outpy.append(hUout);
    outpy.append(n);
    outpy.append(A);
    // Return output as a Python list
    return outpy;
}

template <
    typename XDecimalType,
    typename FDecimalType,
    typename IndexType
>
py::array rf_dl_sg_postproc_mean(
    std::string const &reductionType,
    py::array const &X,
    py::list const &zBatch,
    XDecimalType const cellSize,
    py::list const &A,
    py::list const &n,
    py::list const &h,
    FDecimalType const minClipValue,
    int const nthreads
){
    // Convert
    size_t const bs = py::len(zBatch); // Batch size (number of rfields.)
    vector<arma::Row<XDecimalType>> _A(bs);
    vector<arma::Col<IndexType>> _n(bs);
    vector<map<IndexType, IndexType>> _h(bs);
    vector<arma::Mat<FDecimalType>> _zBatch(bs);
    for(size_t k = 0 ; k < bs ; ++k){
        _A[k] = carma::arr_to_row_view<XDecimalType>(A[k].cast<py::array>());
        _n[k] = carma::arr_to_col_view<IndexType>(n[k].cast<py::array>());
        _h[k] = h[k].cast<py::dict>().cast<map<IndexType, IndexType>>();
        _zBatch[k] = carma::arr_to_mat_view<FDecimalType>(
            zBatch[k].cast<py::array>()
        );
    }
    arma::Mat<XDecimalType> const _X = carma::arr_to_mat_view<XDecimalType>(X);
    // Instantiate post-processor
    rfield::DLSGPostProcessor<
        XDecimalType, FDecimalType, IndexType
    > dlSGPostProc(
        cellSize,
        _A,
        _n,
        _h,
        reductionType,
        minClipValue,
        nthreads
    );
    // Compute post-processed probabilities
    arma::Mat<FDecimalType> out = dlSGPostProc(
        _X,
        _zBatch
    );
    // Return post-processed probabilities
    return carma::mat_to_arr(out, false);
}



}

#endif