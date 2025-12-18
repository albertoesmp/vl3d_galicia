#include <alg/GridMesher.hpp>

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType>
GridMesher<XDecimalType>::GridMesher(
    arma::Col<XDecimalType> const &cellSize,
    bool const extraNodes
) :
    cellSize(cellSize),
    extraNodes(extraNodes)
{}


// ***   GRID METHODS  *** //
// *********************** //
template <typename XDecimalType>
arma::Mat<XDecimalType> GridMesher<XDecimalType>::computeNodes(
    arma::Mat<XDecimalType> const &X
) const{
    arma::Col<XDecimalType> const a = arma::min(X, 0).as_col();
    arma::Col<XDecimalType> const b = arma::max(X, 0).as_col();
    return computeNodes(a, b);
}

template <typename XDecimalType>
arma::Mat<XDecimalType> GridMesher<XDecimalType>::computeNodes(
    arma::Col<XDecimalType> const &a,
    arma::Col<XDecimalType> const &b
) const{
    arma::uword const nx = a.n_rows;
    if(nx == 2) return computeNodes2D(a, b);
    else if(nx == 3)  return computeNodes3D(a, b);
    else return computeNodesND(a, b);
}

template <typename XDecimalType>
arma::Mat<XDecimalType> GridMesher<XDecimalType>::computeNodes2D(
    arma::Col<XDecimalType> const &a,
    arma::Col<XDecimalType> const &b
) const{
    // Prepare computation
    arma::Col<XDecimalType> const d = b-a; // Axis length
    arma::Col<arma::uword> const nodesPerAxis = arma::conv_to<
        arma::Col<arma::uword>
    >::from(arma::ceil(d/cellSize.subvec(0, 1))) +
        (extraNodes ? 1 : 0);
    arma::uword const numCells = arma::prod(nodesPerAxis);
    arma::Mat<XDecimalType> nodes(numCells, 2); // Grid nodes
    // Precompute axis-wise nodes
    arma::Col<XDecimalType> xNodes (nodesPerAxis[0]);
    arma::Col<XDecimalType> yNodes (nodesPerAxis[1]);
    for(arma::uword k = 0 ; k < nodesPerAxis[0] ; ++k){
        xNodes[k] = a[0] + k * cellSize[0];
    }
    for(arma::uword k = 0 ; k < nodesPerAxis[1] ; ++k){
        yNodes[k] = a[1] + k * cellSize[1];
    }
    // Compute all nodes
    arma::uword gidx = 0;
    for(arma::uword i = 0 ; i < nodesPerAxis[0] ; ++i){ // i on x
        for(arma::uword j = 0 ; j < nodesPerAxis[1] ; ++j , ++gidx){ // j on y
            nodes.at(gidx, 0) = xNodes[i];
            nodes.at(gidx, 1) = yNodes[j];
        }
    }
    return nodes;
}

template <typename XDecimalType>
arma::Mat<XDecimalType> GridMesher<XDecimalType>::computeNodes3D(
    arma::Col<XDecimalType> const &a,
    arma::Col<XDecimalType> const &b
) const{
    // Prepare computation
    arma::Col<XDecimalType> const d = b-a; // Axis length
    arma::Col<arma::uword> const nodesPerAxis = arma::conv_to<
        arma::Col<arma::uword>
    >::from(arma::ceil(d/cellSize.subvec(0, 2))) +
        (extraNodes ? 1 : 0);
    arma::uword const numCells = arma::prod(nodesPerAxis);
    arma::Mat<XDecimalType> nodes(numCells, 3); // Grid nodes
    // Precompute axis-wise nodes
    arma::Col<XDecimalType> xNodes (nodesPerAxis[0]);
    arma::Col<XDecimalType> yNodes (nodesPerAxis[1]);
    arma::Col<XDecimalType> zNodes (nodesPerAxis[2]);
    for(arma::uword k = 0 ; k < nodesPerAxis[0] ; ++k){
        xNodes[k] = a[0] + ((XDecimalType)k) * cellSize[0];
    }
    for(arma::uword k = 0 ; k < nodesPerAxis[1] ; ++k){
        yNodes[k] = a[1] + ((XDecimalType)k) * cellSize[1];
    }
    for(arma::uword k = 0 ; k < nodesPerAxis[2] ; ++k){
        zNodes[k] = a[2] + ((XDecimalType)k) * cellSize[2];
    }
    // Compute all nodes
    arma::uword gidx = 0;
    for(arma::uword i = 0 ; i < nodesPerAxis[0] ; ++i){ // i on x
        for(arma::uword j = 0 ; j < nodesPerAxis[1] ; ++j){ // j on y
            for(arma::uword k = 0 ; k < nodesPerAxis[2] ; ++k,++gidx){ // k on z
                nodes.at(gidx, 0) = xNodes[i];
                nodes.at(gidx, 1) = yNodes[j];
                nodes.at(gidx, 2) = zNodes[k];
            }
        }
    }
    // Return nodes
    return nodes;
}

template <typename XDecimalType>
arma::Mat<XDecimalType> GridMesher<XDecimalType>::computeNodesND(
    arma::Col<XDecimalType> const &a,
    arma::Col<XDecimalType> const &b
) const{
    // Prepare computation
    arma::uword const nx = a.n_rows; // Structure space dimensionality
    arma::Col<XDecimalType> const d = arma::ceil(b-a); // Axis length
    arma::Col<arma::uword> const nodesPerAxis = arma::conv_to<
        arma::Col<arma::uword>
    >::from(arma::ceil(d/cellSize.subvec(0, nx-1))) +
        (extraNodes ? 1 : 0);
    arma::uword const numCells = arma::prod(nodesPerAxis);
    arma::Mat<XDecimalType> nodes(numCells, nx); // Grid nodes
    // Precompute axis-wise nodes
    std::vector<arma::Col<XDecimalType>> axisNodes(nx);
    for(arma::uword k = 0 ; k < nx ; ++k){
        axisNodes[k] = arma::Col<XDecimalType>(nodesPerAxis[k]);
        for(arma::uword p = 0 ; p < nodesPerAxis[k] ; ++p){
            axisNodes[k] = a[k] + p * cellSize[k];
        }
    }
    // Compute all nodes
    arma::Col<arma::uword> multiIndex(nx); // Axis-wise indices
    arma::uword const lastAxisIdx = nx-1;
    for(arma::uword gidx = 0 ; gidx < numCells ; ++gidx){ // Iterate over nodes
        // Compute the node index for each axis
        multiIndex[lastAxisIdx] = gidx % nodesPerAxis[lastAxisIdx];
        arma::uword denom = 1; // Denominator for first iteration
        for( // First iteration computed before, last after to avoid one mod (%)
            arma::uword axisIdx = lastAxisIdx-1 ;
            axisIdx > 0 ;
            --axisIdx
        ){ // Iterate over 0 < axisIdx < lastAxisIdx (i.e., from 2nd to prelast)
            denom *= nodesPerAxis[axisIdx-1];
            multiIndex[axisIdx] = static_cast<arma::uword>(std::floor(
                static_cast<double>(gidx) / static_cast<double>(denom)
            )) % nodesPerAxis[axisIdx];
        }
        denom *= nodesPerAxis[1]; // Denominator for last iteration
        multiIndex[0] = static_cast<arma::uword>(std::floor(
            static_cast<double>(gidx) / static_cast<double>(denom)
        ));
        // Compute the coordinates for the node
        for(arma::uword k = 0 ; k < nx ; ++k){
            nodes.at(gidx, k) = a[k] + multiIndex[k] * cellSize[k];
        }
    }
    // Return nodes
    return nodes;
}
