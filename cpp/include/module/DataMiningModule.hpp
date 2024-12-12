/**
 * Provides functions wrapping VL3DPP to be easily called from the VL3D
 * python software.
 * More concretely, the functions here wrap data mining components.
 */

// ***   INCLUDES   *** //
// ******************** //
#include <pcl/pcl_base.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <carma>
#include <armadillo>

#include <string>
#include <vector>

#include <mining/HeightFeatsMiner.hpp>
#include <mining/SmoothFeatsMiner.hpp>
#include <mining/RecountMiner.hpp>

namespace py = pybind11;

namespace vl3dpp::pymods {

// ***   MODULE CALLS   *** //
// ************************ //
template <typename XDecimalType, typename FDecimalType>
py::array mine_smooth_feats(
    std::string const &nbhType,
    int const nbhK,
    py::array const &nbhRadius,
    XDecimalType const nbhLowerBound,
    XDecimalType const nbhUpperBound,
    FDecimalType const weightedMeanOmega,
    FDecimalType const gaussianRbfOmega,
    std::string const &nanPolicy,
    py::list &fnames,
    int const nthreads,
    py::array &X,
    py::array &F
){
    // Convert fnames from Python list of str to C++ vector of string
    std::vector<std::string> _fnames = fnames.cast<std::vector<std::string>>();
    // Return smooth features
    vl3dpp::mining::SmoothFeatsMiner<
        XDecimalType, FDecimalType
    >smoothFeatsMiner(
        nbhType,
        nbhK,
        carma::arr_to_mat_view<XDecimalType>(nbhRadius),
        nbhLowerBound,
        nbhUpperBound,
        weightedMeanOmega,
        gaussianRbfOmega,
        nanPolicy,
        _fnames,
        nthreads
    );
    arma::Mat<FDecimalType> Fhat = smoothFeatsMiner.mine(
        carma::arr_to_mat_view<XDecimalType>(X),
        carma::arr_to_mat_view<FDecimalType>(F)
    );
    return carma::mat_to_arr(Fhat, false);
}

template <typename XDecimalType, typename FDecimalType>
py::array mine_height_feats(
    std::string const &nbhType,
    XDecimalType const nbhRadius,
    XDecimalType const nbhSeparationFactor,
    std::string const &outlierFilter,
    py::list &fnames,
    int const nthreads,
    py::array &X
){
    // Convert fnames from Python list of str to C++ vector of string
    std::vector<std::string> _fnames = fnames.cast<std::vector<std::string>>();
    // Return height features
    vl3dpp::mining::HeightFeatsMiner<
        XDecimalType, FDecimalType
    >heightFeatsMiner(
        nbhType,
        nbhRadius,
        nbhSeparationFactor,
        outlierFilter,
        _fnames,
        nthreads
    );
    arma::Mat<FDecimalType> Fhat = heightFeatsMiner.mine(
        carma::arr_to_mat_view<XDecimalType>(X)
    );
    return carma::mat_to_arr(Fhat, false);
};

template <typename XDecimalType, typename FDecimalType>
py::array mine_recount(
    std::string const &nbhType,
    int const nbhK,
    py::array const &nbhRadius,
    XDecimalType const nbhLowerBound,
    XDecimalType const nbhUpperBound,
    py::list &ignoreNan,
    py::list &absFreq,
    py::list &relFreq,
    py::list &surfDens,
    py::list &volDens,
    py::list &vertSeg,
    py::list &rings,
    py::list &radBound,
    py::list &sect2D,
    py::list &sect3D,
    py::list &condFeatIndices,
    py::list &condTypes,
    py::list &condTargets,
    int const nthreads,
    py::array &X,
    py::array &F
){
    // Convert lists of flags to vector of booleans
    std::vector<bool> _ignoreNan = ignoreNan.cast<std::vector<bool>>();
    std::vector<bool> _absFreq = absFreq.cast<std::vector<bool>>();
    std::vector<bool> _relFreq = relFreq.cast<std::vector<bool>>();
    std::vector<bool> _surfDens = surfDens.cast<std::vector<bool>>();
    std::vector<bool> _volDens = volDens.cast<std::vector<bool>>();
    // Convert lists of integers to vector of integers
    std::vector<int> _vertSeg = vertSeg.cast<std::vector<int>>();
    std::vector<int> _rings = rings.cast<std::vector<int>>();
    std::vector<int> _radBound = radBound.cast<std::vector<int>>();
    std::vector<int> _sect2D = sect2D.cast<std::vector<int>>();
    std::vector<int> _sect3D = sect3D.cast<std::vector<int>>();
    // Convert lists of lists to vectors of vectors
    std::vector<std::vector<arma::uword>> _condFeatIndices =
        condFeatIndices.cast<std::vector<std::vector<arma::uword>>>();
    std::vector<std::vector<std::string>> _condTypes =
        condTypes.cast<std::vector<std::vector<std::string>>>();
    std::vector<std::vector<std::vector<FDecimalType>>> _condTargets =
        condTargets.cast<std::vector<std::vector<std::vector<FDecimalType>>>>();
    // Return recount-based features
    vl3dpp::mining::RecountMiner<
        XDecimalType, FDecimalType
    > recountMiner(
        nbhType,
        nbhK,
        carma::arr_to_mat_view<XDecimalType>(nbhRadius),
        nbhLowerBound,
        nbhUpperBound,
        _ignoreNan,
        _absFreq,
        _relFreq,
        _surfDens,
        _volDens,
        _vertSeg,
        _rings,
        _radBound,
        _sect2D,
        _sect3D,
        _condFeatIndices,
        _condTypes,
        _condTargets,
        nthreads
    );
    arma::Mat<FDecimalType> Fhat = recountMiner.mine(
        carma::arr_to_mat_view<XDecimalType>(X),
        carma::arr_to_mat_view<FDecimalType>(F)
    );
    return carma::mat_to_arr(Fhat, false);
};

}