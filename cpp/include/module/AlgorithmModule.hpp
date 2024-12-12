/**
 * Provides functions wrapping VL3DPP to be easily called from the VL3D
 * python software.
 * More concretely, the functions here wrap algorithms.
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

#include <alg/SupportNeighborhoods.hpp>
#include <alg/Oversampler.hpp>

namespace py = pybind11;
using namespace vl3dpp::alg;

namespace vl3dpp::pymods {


// ***   MODULE CALLS   *** //
// ************************ //
template <
    typename XDecimalType, typename LabelType, typename IndexType
>
py::tuple alg_support_neighborhoods(
    std::string const &nbhType,
    IndexType const nbhK,
    py::array const &nbhRadii,
    XDecimalType const nbhSeparationFactor,
    std::string const &strategy,
    IndexType const numPoints,
    short const fast,
    py::array const &trainingClassDistribution,
    bool const centerOnPcloud,
    bool const extraNodes,
    int const nthreads,
    py::array const &X,
    py::array const &y
){
    // Instantiate algorithm
    SupportNeighborhoods<XDecimalType, LabelType, IndexType> sn(
        nbhType,
        nbhK,
        carma::arr_to_col<XDecimalType>(nbhRadii),
        nbhSeparationFactor,
        strategy,
        numPoints,
        fast,
        carma::arr_to_col_view<IndexType>(trainingClassDistribution),
        centerOnPcloud,
        extraNodes,
        nthreads
    );
    // Compute algorithm
    arma::Mat<XDecimalType> Xsup;
    std::vector<arma::Col<IndexType>> I;
    sn.computeAll(
        carma::arr_to_mat_view<XDecimalType>(X),
        carma::arr_to_col_view<LabelType>(y),
        Xsup,
        I
    );
    // Return to python
    py::list _I;
    for(arma::Col<IndexType> const &Ii : I){
        _I.append(carma::col_to_arr<IndexType>(Ii));
    }
    return py::make_tuple(
        carma::mat_to_arr<XDecimalType>(Xsup),
        _I
    );
}

template <typename XDecimalType, typename IndexType>
py::array alg_oversampler(
    IndexType const minPoints,
    IndexType const targetPoints,
    std::string const& strategy,
    IndexType const K,
    XDecimalType const radius,
    py::array const &X
){
    // Instantiate oversampler
    Oversampler oversampler(
        minPoints,
        targetPoints,
        strategy,
        K,
        radius
    );
    // Return oversampling to python
    return carma::mat_to_arr<XDecimalType>(oversampler(
        carma::arr_to_mat_view<XDecimalType>(X)
    ));
}

}
