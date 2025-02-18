#ifndef _RECEPTIVE_FIELD_MODULE_COMMON_
#define _RECEPTIVE_FIELD_MODULE_COMMON_

/**
* @author Alberto M. Esmoris Pena
*
* Provides common methods for the ReceptiveFieldModule.
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

namespace py = pybind11;
using vl3dpp::alg::SupportNeighborhoods;

namespace vl3dpp::pymods {

// ***  COMMON METHODS  *** //
// ************************ //
template <
    typename XDecimalType,
    typename LabelType,
    typename IndexType
>
SupportNeighborhoods<XDecimalType, LabelType, IndexType>
instantiateSupportNeighborhoods(
    py::list const& supportArgs,
    int const nthreads
){
    // Extract instantiation arguments from supportArgs list
    std::string const nbhType = supportArgs[0].cast<std::string>();
    IndexType const nbhK = supportArgs[1].cast<IndexType>();
    arma::Col<XDecimalType> const nbhRadii = carma::arr_to_col_view<
        XDecimalType
    >(supportArgs[2].cast<py::array>());
    XDecimalType const nbhSeparationFactor = supportArgs[3].cast<
        XDecimalType
    >();
    std::string const strategy = supportArgs[4].cast<std::string>();
    IndexType const numPoints = supportArgs[5].cast<IndexType>();
    short const supportFast = supportArgs[6].cast<short>();
    arma::Col<IndexType> const trainingClassDistribution =
        carma::arr_to_col_view<IndexType>(
            supportArgs[7].cast<py::array>()
        );
    bool const centerOnPcloud = supportArgs[8].cast<bool>();
    bool const extraNodes = supportArgs[9].cast<bool>();
    // Build and return support neighborhoods instance
    return SupportNeighborhoods<XDecimalType, LabelType, IndexType>(
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
}

}

#endif