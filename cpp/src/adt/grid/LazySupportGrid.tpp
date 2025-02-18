// ***   INCLUDES   *** //
// ******************** //
#include <adt/grid/LazySupportGrid.hpp>
#include <util/VL3DPPException.hpp>
#include <sstream>

using vl3dpp::util::VL3DPPException;

namespace vl3dpp::adt::grid {

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename DecimalType>
LazySupportGrid<DecimalType>::LazySupportGrid(
    arma::Col<DecimalType> const &halfLength,
    arma::Col<DecimalType> const &xmin,
    arma::Col<DecimalType> const &xmax
) :
    halfLength(halfLength),
    xmin(xmin),
    xmax(xmax)
{
    // Validate dimensionality
    if(xmin.n_elem != xmax.n_elem){
        std::stringstream ss;
        ss  << "LazySupportGrid could not be instantiated because "
            << "the dimensionality of the min point is " << xmin.n_elem
            << " and the dimensionality of the max point is "
            << xmax.n_elem << ". "
            << "These dimensionalities MUST be the same.";
        throw VL3DPPException(ss.str());
    }
    if(xmin.n_elem != halfLength.n_elem){
        std::stringstream ss;
        ss  << "LazySupportGrid could not be instantiated because "
            << "the dimensionality of the min point is " << xmin.n_elem
            << " and the dimensionality of the max point is "
            << halfLength.n_elem << ". "
            << "These dimensionalities MUST be the same.";
        throw VL3DPPException(ss.str());
    }
    // Find number of cells per axis
    n = arma::conv_to<arma::Col<arma::uword>>::from(
        arma::ceil( (xmax-xmin) / (2*halfLength) )
    );
}

template <typename DecimalType>
LazySupportGrid<DecimalType>::LazySupportGrid(
    arma::Col<DecimalType> const &halfLength,
    arma::Mat<DecimalType> const &X
) :
    LazySupportGrid(
        halfLength,
        arma::min(X, 0).as_col(),
        arma::max(X, 0).as_col()
    )
{}


// ***   GRID METHODS   *** //
// ************************ //
template <typename DecimalType>
arma::Col<DecimalType> LazySupportGrid<DecimalType>::getCentroid(
    size_t const phi
) const {
    // Extract indices from index
    arma::Col<arma::uword> i(n.n_elem);
    i[0] = phi % n[0];
    arma::uword indexingFactor = n[0];
    for(arma::uword j=1 ; j < n.n_elem ; ++j){
        i[j] = (int) (phi/indexingFactor) % n[j];
        indexingFactor *= n[j];
    }
    // Return centroid from indices
    return getCentroid(i);
}

template <typename DecimalType>
arma::Col<DecimalType> LazySupportGrid<DecimalType>::getCentroid(
    arma::Col<arma::uword> const i
) const {
    // (2*i+1) for centered centroids, (2*i) for min centroids
    return xmin + ((2*i+1) % halfLength);
}

template <typename DecimalType>
arma::uword LazySupportGrid<DecimalType>::getNumCells() const{
    arma::uword numCells = 1;
    for(arma::uword k = 0 ; k < n.n_elem ; ++k) numCells *= n[k];
    return numCells;
}

template <typename DecimalType>
arma::uword LazySupportGrid<DecimalType>::findCellIndex(
    arma::Col<DecimalType> const &x
) const{
    // Find axis-wise indices
    arma::Col<arma::uword> const i = arma::min(
        arma::conv_to<arma::Col<arma::uword>>::from(
            arma::floor((x-xmin)/(2*halfLength))
        ),
        n-1
    );
    // Find cell index
    arma::uword phi = i[0];
    arma::uword indexingFactor = n[0];
    for(arma::uword k = 1 ; k < i.n_elem ; ++k){
        phi += i[k] * indexingFactor;
        indexingFactor *= n[k];
    }
    return phi;
}



} // Closing namespace