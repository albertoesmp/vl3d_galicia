#include <adt/octree/Octree.hpp>

namespace vl3dpp::adt::octree {

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename IndexType, typename DecimalType>
Octree<IndexType, DecimalType>::Octree(
    arma::Mat<DecimalType> const &X,
    bool const make2D,
    bool const make3D,
    bool const directResolution,
    bool const forceEpnRatio,
    DecimalType const _resolution,
    arma::uword const _maxNodes,
    arma::uword const epnRatio
){
    // Determine resolution : Baseline (default resolution)
    DecimalType resolution = _resolution;
    if(!directResolution) {
        // Determine resolution : Max nodes from epnRatio, if requested
        arma::uword maxNodes = (forceEpnRatio) ?
            std::min((arma::uword) std::ceil(X.n_rows/epnRatio), _maxNodes) :
            _maxNodes;
        // Determine resolution from max nodes, if requested
        DecimalType const d = arma::max(arma::max(X, 0)-arma::min(X, 0));
        resolution = d/std::ceil(std::log2(maxNodes)/3.0);
    }
    // Build the octree
    buildOctree(X, make2D, make3D, resolution);
}

template <typename IndexType, typename DecimalType>
Octree<IndexType, DecimalType>::Octree(
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud3D,
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud2D,
    DecimalType const resolution
){
    // Build 2D octree if 2D pcloud is available
    if(pcloud2D != nullptr){
        this->pcloud2D = pcloud2D;
        octree2D = new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(
            resolution
        );
        buildOctree2D(resolution);
    }
    // Build 3D octree if 3D pcloud is available
    if(pcloud3D != nullptr){
        this->pcloud3D = pcloud3D;
        octree3D = new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(
            resolution
        );
        buildOctree3D(resolution);
    }
}

template <typename IndexType, typename DecimalType>
Octree<IndexType, DecimalType>::~Octree(){
    // Release octrees
    delete octree3D;
    octree3D = nullptr;
    delete octree2D;
    octree2D = nullptr;
}


// ***  ASSIGNMENT OPERATOR  *** //
// ***************************** //
template <typename IndexType, typename DecimalType>
Octree<IndexType, DecimalType> &
Octree<IndexType, DecimalType>::operator=(Octree &&ot){
    // Cannibalize internal Octrees
    delete octree2D;
    octree2D = ot.octree2D;
    ot.octree2D = nullptr;
    delete octree3D;
    octree3D = ot.octree3D;
    ot.octree3D = nullptr;
    // Cannibalize internal point clouds
    pcloud2D = ot.pcloud2D;
    pcloud3D = ot.pcloud3D;
    // Return reference to this instance
    return *this;
}


// ***   BUILDING METHODS   *** //
// **************************** //
template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::buildOctree(
    arma::Mat<DecimalType> const &X,
    bool const make2D,
    bool const make3D,
    DecimalType const resolution
){
    // Build 2D point cloud and octree from given structure space (X)
    if(make2D){
        // Build the 2D point cloud
        pcloud2D = pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>()
        );
        pcloud2D->points.resize(X.n_rows);
        pcl::PointCloud<pcl::PointXYZ> &_pcloud2D = *pcloud2D;
        for(IndexType i = 0 ; i < X.n_rows ; ++i){
            _pcloud2D[i].x = X.at(i, 0);
            _pcloud2D[i].y = X.at(i, 1);
            _pcloud2D[i].z = 0;
        }
        // Build the 2D octree
        buildOctree2D(resolution);
    }
    // Build 3D point cloud and octree from given structure space (X)
    if(make3D) {
        // Build the 3D point cloud
        pcloud3D = pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>()
        );
        pcloud3D->points.resize(X.n_rows);
        pcl::PointCloud<pcl::PointXYZ> &_pcloud3D = *pcloud3D;
        for(IndexType i = 0 ; i < X.n_rows ; ++i){
            _pcloud3D[i].x = X.at(i, 0);
            _pcloud3D[i].y = X.at(i, 1);
            _pcloud3D[i].z = X.at(i, 2);
        }
        // Build the 3D octree
        buildOctree3D(resolution);
    }
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::buildOctree2D(
    DecimalType const resolution
){
    octree2D = new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(
        resolution
    );
    octree2D->setInputCloud(pcloud2D);
    octree2D->addPointsFromInputCloud();
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::buildOctree3D(
    DecimalType const resolution
){
    octree3D = new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(
        resolution
    );
    octree3D->setInputCloud(pcloud3D);
    octree3D->addPointsFromInputCloud();
}




// ***   KNN METHODS   *** //
// *********************** //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findKnn(
    PointType const &x, IndexType const k
){
    arma::Col<IndexType> n(k);
    arma::Col<DecimalType> d(k);
    findKnn(x, k, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findKnn(
    PointType const &x,
    IndexType const k,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    pcl::PointXYZ _x(x[0], x[1], x[2]);
    std::vector<int> _n(k);
    std::vector<float> _d(k);
    octree3D->nearestKSearch(_x, k, _n, _d);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
    d = arma::conv_to<arma::Col<DecimalType>>::from(_d);
}

template <typename IndexType, typename DecimalType>
arma::Mat<IndexType> Octree<IndexType, DecimalType>::findKnn(
    arma::Mat<DecimalType> const &X, IndexType const k
){
    arma::Mat<IndexType> N(X.n_rows, k);
    findKnn(X, k, N);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findKnn(
    arma::Mat<DecimalType> const &X,
    IndexType const k,
    arma::Mat<IndexType> &N,
    arma::Mat<DecimalType> &D
){
    IndexType const m = X.n_rows; // Num points
    arma::Col<IndexType> ni(k);
    arma::Col<DecimalType> di(k);
    for(IndexType i=0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findKnn(xi, k, ni, di); // Find knn neighborhood of i-th point
        N.row(i) = ni.as_row();
        D.row(i) = di.as_row();
    }
}




// ***   2D KNN METHODS   *** //
// ************************** //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findKnn2D(
    PointType const &x, IndexType const k
){
    arma::Col<IndexType> n(k);
    arma::Col<DecimalType> d(k);
    findKnn2D(x, k, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findKnn2D(
    PointType const &x,
    IndexType const k,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    pcl::PointXYZ _x(x[0], x[1], 0.0);
    std::vector<int> _n(k);
    std::vector<float> _d(k);
    octree2D->nearestKSearch(_x, k, _n, _d);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
    d = arma::conv_to<arma::Col<DecimalType>>::from(_d);
}

template <typename IndexType, typename DecimalType>
arma::Mat<IndexType> Octree<IndexType, DecimalType>::findKnn2D(
    arma::Mat<DecimalType> const &X, IndexType const k
){
    arma::Mat<IndexType> N(X.n_rows, k);
    arma::Mat<DecimalType> D(X.n_rows, k);
    findKnn2D(X, k, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findKnn2D(
    arma::Mat<DecimalType> const &X,
    IndexType const k,
    arma::Mat<IndexType> &N,
    arma::Mat<DecimalType> &D
){
    IndexType const m = X.n_rows; // Num points
    arma::Col<IndexType> ni(k);
    arma::Col<DecimalType> di(k);
    for(IndexType i=0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findKnn2D(xi, k, ni, di); // Find knn neighborhood of i-th point
        N.row(i) = ni.as_row();
        D.row(i) = di.as_row();
    }
}




// ***   BOUNDED KNN METHODS   *** //
// ******************************* //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findBoundedKnn(
    PointType const &x,
    IndexType const k,
    DecimalType const maxSquaredDistance
){
    arma::Col<IndexType> n(k);
    arma::Col<DecimalType> d(k);
    findBoundedKnn(x, k, maxSquaredDistance, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findBoundedKnn(
    PointType const &x,
    IndexType const k,
    DecimalType const maxSquaredDistance,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    // Obtain knn
    findKnn(x, k, n, d);
    // Discard out of bounds neighbors
    for(IndexType i = 0 ; i < n.n_elem ; ++i){
        if(d[i] > maxSquaredDistance){
            n = n.rows(0, i-1);
            d = d.rows(0, i-1);
            break;
        }
    }
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>>
Octree<IndexType, DecimalType>::findBoundedKnn(
    arma::Mat<DecimalType> const &X,
    IndexType const k,
    DecimalType const maxSquaredDistance
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    std::vector<arma::Col<DecimalType>> D(X.n_rows);
    findBoundedKnn(X, k, maxSquaredDistance, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findBoundedKnn(
    arma::Mat<DecimalType> const &X,
    IndexType const k,
    DecimalType const maxSquaredDistance,
    std::vector<arma::Col<IndexType>> &N,
    std::vector<arma::Col<DecimalType>> &D
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // Num points
    if(N.size() != m) N.resize(m);
    if(D.size() != m) D.resize(m);
    // Extract bounded KNN
    arma::Col<IndexType> ni(k);
    arma::Col<DecimalType> di(k);
    for(IndexType i=0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findBoundedKnn( // Find i-th bounded knn neighborhood
            xi, k, maxSquaredDistance, ni, di
        );
        N[i] = ni;
        D[i] = di;
    }
}




// ***   BOUNDED 2D KNN METHODS   *** //
// ********************************** //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findBoundedKnn2D(
    PointType const &x,
    IndexType const k,
    DecimalType const maxSquaredDistance
){
    arma::Col<IndexType> n(k);
    arma::Col<DecimalType> d(k);
    findBoundedKnn2D(x, k, maxSquaredDistance, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findBoundedKnn2D(
    PointType const &x,
    IndexType const k,
    DecimalType const maxSquaredDistance,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    // Obtain 2D knn
    findKnn2D(x, k, n, d);
    // Discard out of bounds neighbors
    for(IndexType i = 0 ; i < n.n_elem ; ++i){
        if(d[i] > maxSquaredDistance){
            n = n.rows(0, i-1);
            d = d.rows(0, i-1);
            break;
        }
    }
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>>
Octree<IndexType, DecimalType>::findBoundedKnn2D(
    arma::Mat<DecimalType> const &X,
    IndexType const k,
    DecimalType const maxSquaredDistance
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    std::vector<arma::Col<DecimalType>> D(X.n_rows);
    findBoundedKnn2D(X, k, maxSquaredDistance, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findBoundedKnn2D(
    arma::Mat<DecimalType> const &X,
    IndexType const k,
    DecimalType const maxSquaredDistance,
    std::vector<arma::Col<IndexType>> &N,
    std::vector<arma::Col<DecimalType>> &D
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // num points
    if(N.size() != m) N.resize(m);
    if(D.size() != m) D.resize(m);
    // Extract bounded 2D KNN
    arma::Col<IndexType> ni(k);
    arma::Col<DecimalType> di(k);
    for(IndexType i=0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findBoundedKnn2D( // Find i-th bounded 2D knn neighborhood
            xi, k, maxSquaredDistance, ni, di
        );
        N[i] = ni;
        D[i] = di;
    }
}




// ***  SPHERICAL NEIGHBORHOOD METHODS  *** //
// **************************************** //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findSphere(
    PointType const &x,
    DecimalType const r
){
    arma::Col<IndexType> n;
    arma::Col<DecimalType> d;
    findSphere(x, r, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findSphere(
    PointType const &x,
    DecimalType const r,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    pcl::PointXYZ _x(x[0], x[1], x[2]);
    std::vector<int> _n;
    std::vector<float> _d;
    octree3D->radiusSearch(_x, r, _n, _d);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
    d = arma::conv_to<arma::Col<DecimalType>>::from(_d);
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>> Octree<IndexType, DecimalType>::findSphere(
    arma::Mat<DecimalType> const &X,
    DecimalType const r
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    std::vector<arma::Col<DecimalType>> D(X.n_rows);
    findSphere(X, r, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findSphere(
    arma::Mat<DecimalType> const &X,
    DecimalType const r,
    std::vector<arma::Col<IndexType>> &N,
    std::vector<arma::Col<DecimalType>> &D
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // Num points
    if(N.size() != m) N.resize(m);
    if(D.size() != m) D.resize(m);
    // Extract spherical neighborhood
    arma::Col<IndexType> ni;
    arma::Col<DecimalType> di;
    for(IndexType i = 0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findSphere(xi, r, ni, di);
        N[i] = ni;
        D[i] = di;
    }
}




// ***  CYLINDRICAL NEIGHBORHOOD METHODS  *** //
// ****************************************** //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findCylinder(
    PointType const &x,
    DecimalType const r
){
    arma::Col<IndexType> n;
    arma::Col<DecimalType> d;
    findCylinder(x, r, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findCylinder(
    PointType const &x,
    DecimalType const r,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    pcl::PointXYZ _x(x[0], x[1], 0);
    std::vector<int> _n;
    std::vector<float> _d;
    octree2D->radiusSearch(_x, r, _n, _d);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
    d = arma::conv_to<arma::Col<DecimalType>>::from(_d);
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>> Octree<IndexType, DecimalType>::findCylinder(
    arma::Mat<DecimalType> const &X,
    DecimalType const r
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    std::vector<arma::Col<DecimalType>> D(X.n_rows);
    findCylinder(X, r, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findCylinder(
    arma::Mat<DecimalType> const &X,
    DecimalType const r,
    std::vector<arma::Col<IndexType>> &N,
    std::vector<arma::Col<DecimalType>> &D
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // Num points
    if(N.size() != m) N.resize(m);
    if(D.size() != m) D.resize(m);
    // Extract cylindrical neighborhood
    arma::Col<IndexType> ni;
    arma::Col<DecimalType> di;
    for(IndexType i = 0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findCylinder(xi, r, ni, di);
        N[i] = ni;
        D[i] = di;
    }
}




// ***  BOUNDED CYLINDRICAL NEIGHBORHOOD METHODS  *** //
// ************************************************** //
template <typename IndexType, typename DecimalType>
template <typename PointType, typename BoundaryCoordinateType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findBoundedCylinder(
    PointType const &x,
    BoundaryCoordinateType const &z,
    DecimalType const r,
    DecimalType const zmin,
    DecimalType const zmax
){
    arma::Col<IndexType> n;
    arma::Col<DecimalType> d;
    findBoundedCylinder(x, z, r, zmin, zmax, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType, typename BoundaryCoordinateType>
void Octree<IndexType, DecimalType>::findBoundedCylinder(
    PointType const &x,
    BoundaryCoordinateType const &z,
    DecimalType const r,
    DecimalType const zmin,
    DecimalType const zmax,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    // Obtain cylinder
    findCylinder(x, r, n ,d);
    // Discard out of bounds neighbors
    std::vector<arma::uword> remove; // Indices of points to be removed
    remove.reserve(n.n_elem);
    double const zi = x[2];
    for(IndexType j = 0 ; j < n.n_elem ; ++j){ // Find points to be removed
        double const zj = z[n[j]] - zi;
        if(zj < zmin || zj > zmax) remove.push_back(j);
    }
    // Remove the points
    n.shed_rows(arma::conv_to<arma::Col<arma::uword>>::from(remove));
}

template <typename IndexType, typename DecimalType>
template <typename PointType, typename BoundaryCoordinateType>
std::vector<arma::Col<IndexType>>
Octree<IndexType, DecimalType>::findBoundedCylinder(
    arma::Mat<DecimalType> const &X,
    BoundaryCoordinateType const &z,
    DecimalType const r,
    DecimalType const zmin,
    DecimalType const zmax
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    std::vector<arma::Col<DecimalType>> D(X.n_rows);
    findBoundedCylinder(X, z, r, zmin, zmax, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
template <typename PointType, typename BoundaryCoordinateType>
void Octree<IndexType, DecimalType>::findBoundedCylinder(
    arma::Mat<DecimalType> const &X,
    BoundaryCoordinateType const &z,
    DecimalType const r,
    DecimalType const zmin,
    DecimalType const zmax,
    std::vector<arma::Col<IndexType>> &N,
    std::vector<arma::Col<DecimalType>> &D
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // num points
    if(N.size() != m) N.resize(m);
    if(D.size() != m) D.resize(m);
    // Extract bounded cylinder
    arma::Col<IndexType> ni;
    arma::Col<DecimalType> di;
    for(IndexType i=0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findBoundedCylinder( // Find i-th bounded cylindrical neighborhood
            xi, z, r, zmin, zmax, ni, di
        );
        N[i] = ni;
        D[i] = di;
    }
}




// ***  BOX NEIGHBORHOOD METHODS  *** //
// ********************************** //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findBox(
    PointType const &x,
    PointType const halfLength
){
    arma::Col<IndexType> n;
    findBox(x, halfLength, n);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findBox(
    PointType const &x,
    PointType const halfLength,
    arma::Col<IndexType> &n
){
    // Bounding box min vertex
    Eigen::Vector3f const a(
        x[0]-halfLength[0], x[1]-halfLength[1], x[2]-halfLength[2]
    );
    // Bounding box max vertex
    Eigen::Vector3f const b(
        x[0]+halfLength[0], x[1]+halfLength[1], x[2]+halfLength[2]
    );
    // Extract box-like neighborhood
    std::vector<int> _n;
    octree3D->boxSearch(a, b, _n);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>> Octree<IndexType, DecimalType>::findBox(
    arma::Mat<DecimalType> const &X,
    arma::Col<DecimalType> const halfLength
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    findBox(X, halfLength, N);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findBox(
    arma::Mat<DecimalType> const &X,
    arma::Col<DecimalType> const halfLength,
    std::vector<arma::Col<IndexType>> &N
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // Num points
    if(N.size() != m) N.resize(m);
    // Extract spherical neighborhood
    arma::Col<IndexType> ni;
    for(IndexType i = 0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findBox(xi, halfLength, ni);
        N[i] = ni;
    }
}




// ***  RECTANGULAR NEIGHBORHOOD METHODS  *** //
// ****************************************** //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findRectangle(
    PointType const &x,
    PointType const halfLength
){
    arma::Col<IndexType> n;
    findRectangle(x, halfLength, n);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findRectangle(
    PointType const &x,
    PointType const halfLength,
    arma::Col<IndexType> &n
){
    // Bounding box min vertex
    Eigen::Vector3f const a(
        x[0]-halfLength[0], x[1]-halfLength[1], 0.0
    );
    // Bounding box max vertex
    Eigen::Vector3f const b(
        x[0]+halfLength[0], x[1]+halfLength[1], 0.0
    );
    // Extract box-like neighborhood
    std::vector<int> _n;
    octree2D->boxSearch(a, b, _n);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>>
Octree<IndexType, DecimalType>::findRectangle(
    arma::Mat<DecimalType> const &X,
    arma::Col<DecimalType> const halfLength
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    findRectangle(X, halfLength, N);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findRectangle(
    arma::Mat<DecimalType> const &X,
    arma::Col<DecimalType> const halfLength,
    std::vector<arma::Col<IndexType>> &N
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // Num points
    if(N.size() != m) N.resize(m);
    // Extract spherical neighborhood
    arma::Col<IndexType> ni;
    for(IndexType i = 0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findRectangle(xi, halfLength, ni);
        N[i] = ni;
    }
}




// ***  BOUNDED RECTANGULAR NEIGHBORHOOD METHODS  *** //
// ************************************************** //
template <typename IndexType, typename DecimalType>
template <typename PointType, typename BoundaryCoordinateType>
arma::Col<IndexType> Octree<IndexType, DecimalType>::findBoundedRectangle(
    PointType const &x,
    BoundaryCoordinateType const &z,
    PointType const halfLength,
    DecimalType const zmin,
    DecimalType const zmax
){
    arma::Col<IndexType> n;
    findBoundedRectangle(x, z, halfLength, zmin, zmax, n);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType, typename BoundaryCoordinateType>
void Octree<IndexType, DecimalType>::findBoundedRectangle(
    PointType const &x,
    BoundaryCoordinateType const &z,
    PointType const halfLength,
    DecimalType const zmin,
    DecimalType const zmax,
    arma::Col<IndexType> &n
){
    // Obtain rectangle
    findRectangle(x, halfLength, n);
    // Discard out of bounds neighbors
    std::vector<arma::uword> remove; // Indices of points to be removed
    remove.reserve(n.n_elem);
    double const zi = x[2];
    for(IndexType j = 0 ; j < n.n_elem ; ++j){ // Find points to be removed
        double const zj = z[n[j]] - zi;
        if(zj < zmin || zj > zmax) remove.push_back(j);
    }
    // Remove the points
    n.shed_rows(arma::conv_to<arma::Col<arma::uword>>::from(remove));
}

template <typename IndexType, typename DecimalType>
template <typename BoundaryCoordinateType>
std::vector<arma::Col<IndexType>>
Octree<IndexType, DecimalType>::findBoundedRectangle(
    arma::Mat<DecimalType> const &X,
    BoundaryCoordinateType const &z,
    arma::Col<DecimalType> const halfLength,
    DecimalType const zmin,
    DecimalType const zmax
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    findBoundedRectangle(X, z, halfLength, zmin, zmax, N);
    return N;
}

template <typename IndexType, typename DecimalType>
template <typename BoundaryCoordinateType>
void Octree<IndexType, DecimalType>::findBoundedRectangle(
    arma::Mat<DecimalType> const &X,
    BoundaryCoordinateType const &z,
    arma::Col<DecimalType> const halfLength,
    DecimalType const zmin,
    DecimalType const zmax,
    std::vector<arma::Col<IndexType>> &N
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // num points
    if(N.size() != m) N.resize(m);
    // Extract bounded cylinder
    arma::Col<IndexType> ni;
    for(IndexType i=0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findBoundedRectangle( // Find i-th bounded rectangular neighborhood
            xi, z, halfLength, zmin, zmax, ni
        );
        N[i] = ni;
    }
}




// ***  OCTANT NEIGHBORHOOD METHODS  *** //
// ************************************* //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType>
Octree<IndexType, DecimalType>::findOctant(PointType const &x){
    arma::Col<IndexType> n;
    findOctant(x, n);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void Octree<IndexType, DecimalType>::findOctant(
    PointType const &x, arma::Col<IndexType> &n
){
    pcl::PointXYZ _x(x[0], x[1], x[2]);
    std::vector<int> _n;
    octree3D->voxelSearch(_x, _n);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>> Octree<IndexType, DecimalType>::findOctant(
    arma::Mat<DecimalType> const &X
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    findOctant(X, N);
    return N;
}

template <typename IndexType, typename DecimalType>
void Octree<IndexType, DecimalType>::findOctant(
    arma::Mat<DecimalType> const &X,
    std::vector<arma::Col<IndexType>> &N
){
    // Force correctness of output dimensionality
    IndexType const m = X.n_rows; // Num points
    if(N.size() != m) N.resize(m);
    // Extract octant neighborhood
    arma::Col<IndexType> ni;
    for(IndexType i = 0 ; i < m ; ++i){ // For each i-th point
        arma::Col<DecimalType> const xi = X.row(i).as_col();
        findOctant(xi, ni);
        N[i] = ni;
    }
}

}