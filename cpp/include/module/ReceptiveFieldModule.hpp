/**
 * Provides functions wrapping VL3DPP to be easily called from the VL3D
 * python software.
 * More concretely, the functions here wrap receptive field components.
 */

// ***   INCLUDES   *** //
// ******************** //
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
#include <alg/Oversampler.hpp>
#include <rfield/DLPreProcessorOutput.hpp>
#include <rfield/DLFPSPostProcessor.hpp>
#include <util/VL3DPPMacros.hpp>

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
    std::string const nbhType = supportArgs[0].cast<std::string>();
    InternalIndexType const nbhK = supportArgs[1].cast<InternalIndexType>();
    arma::Col<InputXDecimalType> const nbhRadii = carma::arr_to_col_view<
        InputXDecimalType
    >(supportArgs[2].cast<py::array>());
    InputXDecimalType const nbhSeparationFactor = supportArgs[3].cast<
        InputXDecimalType
    >();
    std::string const strategy = supportArgs[4].cast<std::string>();
    InternalIndexType const numPoints = supportArgs[5].cast<InternalIndexType>();
    short const supportFast = supportArgs[6].cast<short>();
    arma::Col<InternalIndexType> const trainingClassDistribution =
        carma::arr_to_col_view<InternalIndexType>(
            supportArgs[7].cast<py::array>()
        );
    bool const centerOnPcloud = supportArgs[8].cast<bool>();
    bool const extraNodes = supportArgs[9].cast<bool>();
    SupportNeighborhoods<InputXDecimalType, LabelType, InternalIndexType> sn(
        nbhType,
        nbhK,
        nbhRadii,
        nbhSeparationFactor,
        strategy,
        numPoints,
        supportFast,
        trainingClassDistribution,
        centerOnPcloud,
        extraNodes,
        nthreads
    );
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
    int nthreads
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
    std::string const nbhType = supportArgs[0].cast<std::string>();
    InternalIndexType const nbhK = supportArgs[1].cast<InternalIndexType>();
    arma::Col<InputXDecimalType> const nbhRadii = carma::arr_to_col_view<
        InputXDecimalType
    >(supportArgs[2].cast<py::array>());
    InputXDecimalType const nbhSeparationFactor = supportArgs[3].cast<
        InputXDecimalType
    >();
    std::string const strategy = supportArgs[4].cast<std::string>();
    InternalIndexType const numPoints = supportArgs[5].cast<InternalIndexType>();
    short const supportFast = supportArgs[6].cast<short>();
    arma::Col<InternalIndexType> const trainingClassDistribution =
        carma::arr_to_col_view<InternalIndexType>(
            supportArgs[7].cast<py::array>()
        );
    bool const centerOnPcloud = supportArgs[8].cast<bool>();
    bool const extraNodes = supportArgs[9].cast<bool>();
    SupportNeighborhoods<InputXDecimalType, LabelType, InternalIndexType> sn(
        nbhType,
        nbhK,
        nbhRadii,
        nbhSeparationFactor,
        strategy,
        numPoints,
        supportFast,
        trainingClassDistribution,
        centerOnPcloud,
        extraNodes,
        nthreads
    );
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

}
