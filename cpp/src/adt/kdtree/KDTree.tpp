#include <adt/kdtree/KDTree.hpp>

namespace vl3dpp::adt::kdtree {

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename IndexType, typename DecimalType>
KDTree<IndexType, DecimalType>::KDTree(
    arma::Mat<DecimalType> const &X,
    bool const make2D,
    bool const make3D
){
    // Build 2D point cloud and KDTree from given structure space (X)
    if(make2D){
        // Build the 2D point cloud
        pcloud2D = pcl::PointCloud<pcl::PointXY>::Ptr(
            new pcl::PointCloud<pcl::PointXY>()
        );
        pcloud2D->points.resize(X.n_rows);
        pcl::PointCloud<pcl::PointXY> &_pcloud2D = *pcloud2D;
        for(IndexType i = 0 ; i < X.n_rows ; ++i){
            _pcloud2D[i].x = X.at(i, 0);
            _pcloud2D[i].y = X.at(i, 1);
        }
        // Build the 2D KDTree
        buildKDTree2D();
    }
    // Build 3D point cloud and KDTree from given structure space (X)
    if(make3D){
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
        // Build the 3D KDtree
        buildKDTree3D();
    }
}

template <typename IndexType, typename DecimalType>
KDTree<IndexType, DecimalType>::KDTree(
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud3D,
    pcl::PointCloud<pcl::PointXY>::Ptr pcloud2D
){
    // Build 2D KDTree if 2D pcloud is available
    if(pcloud2D != nullptr){
        this->pcloud2D = pcloud2D;
        buildKDTree2D();
    }
    // Build 3D KDTree if 3D pcloud is available
    if(pcloud3D != nullptr){
        this->pcloud3D = pcloud3D;
        buildKDTree3D();
    }
}

template <typename IndexType, typename DecimalType>
KDTree<IndexType, DecimalType>::~KDTree(){
    // Release kdtrees
    delete kdt2D;
    kdt2D = nullptr;
    delete kdt3D;
    kdt3D = nullptr;
}


// ***  ASSIGNMENT OPERATOR  *** //
// ***************************** //
template <typename IndexType, typename DecimalType>
KDTree<IndexType, DecimalType> &
KDTree<IndexType, DecimalType>::operator=(
    KDTree<IndexType, DecimalType> &&kdt
){
    // Canibalize internal KDTrees
    delete kdt2D;
    kdt2D = kdt.kdt2D;
    kdt.kdt2D = nullptr;
    delete kdt3D;
    kdt3D = kdt.kdt3D;
    kdt.kdt3D = nullptr;
    // Canibalize internal point clouds
    pcloud2D = kdt.pcloud2D;
    pcloud3D = kdt.pcloud3D;
    // Return reference to this instance
    return *this;
}



// ***   BUILDING METHODS   *** //
// **************************** //
template <typename IndexType, typename DecimalType>
void KDTree<IndexType, DecimalType>::buildKDTree2D(){
    kdt2D = new pcl::search::KdTree<pcl::PointXY>();
    kdt2D->setInputCloud(this->pcloud2D);
}

template <typename IndexType, typename DecimalType>
void KDTree<IndexType, DecimalType>::buildKDTree3D(){
    kdt3D = new pcl::search::KdTree<pcl::PointXYZ>();
    kdt3D->setInputCloud(this->pcloud3D);
}




// ***   KNN METHODS   *** //
// *********************** //
template <typename IndexType, typename DecimalType>
template <typename PointType>
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findKnn(
    PointType const &x, IndexType const k
){
    arma::Col<IndexType> n(k);
    arma::Col<DecimalType> d(k);
    findKnn(x, k, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void KDTree<IndexType, DecimalType>::findKnn(
    PointType const &x,
    IndexType const k,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    pcl::PointXYZ _x(x[0], x[1], x[2]);
    std::vector<int> _n(k);
    std::vector<float> _d(k);
    kdt3D->nearestKSearch(_x, k, _n, _d);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
    d = arma::conv_to<arma::Col<DecimalType>>::from(_d);
}

template <typename IndexType, typename DecimalType>
arma::Mat<IndexType> KDTree<IndexType, DecimalType>::findKnn(
    arma::Mat<DecimalType> const &X, IndexType const k
){
    arma::Mat<IndexType> N(X.n_rows, k);
    arma::Mat<DecimalType> D(X.n_rows, k);
    findKnn(X, k, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void KDTree<IndexType, DecimalType>::findKnn(
    arma::Mat<DecimalType> const &X,
    IndexType const k,
    arma::Mat<IndexType> &N,
    arma::Mat<DecimalType> &D
){
    IndexType const m = X.n_rows; // Num points
    arma::Col<IndexType> ni(k);
    arma::Col<DecimalType> di(k);
    for(IndexType i = 0 ; i < m ; ++i){ // For each i-th point
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findKnn2D(
    PointType const &x, IndexType const k
) {
    arma::Col<IndexType> n(k);
    arma::Col<DecimalType> d(k);
    findKnn2D(x, k, n, d);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void KDTree<IndexType, DecimalType>::findKnn2D(
    PointType const &x,
    IndexType const k,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
) {
    pcl::PointXY _x(x[0], x[1]);
    std::vector<int> _n(k);
    std::vector<float> _d(k);
    kdt2D->nearestKSearch(_x, k, _n, _d);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
    d = arma::conv_to<arma::Col<DecimalType>>::from(_d);
}

template <typename IndexType, typename DecimalType>
arma::Mat<IndexType> KDTree<IndexType, DecimalType>::findKnn2D(
    arma::Mat<DecimalType> const &X, IndexType const k
) {
    arma::Mat<IndexType> N(X.n_rows, k);
    arma::Mat<DecimalType> D(X.n_rows, k);
    findKnn2D(X, k, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void KDTree<IndexType, DecimalType>::findKnn2D(
    arma::Mat<DecimalType> const &X,
    IndexType const k,
    arma::Mat<IndexType> &N,
    arma::Mat<DecimalType> &D
) {
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findBoundedKnn(
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
void KDTree<IndexType, DecimalType>::findBoundedKnn(
    PointType const &x,
    IndexType const k,
    DecimalType const maxSquaredDistance,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    // Obtain knn
    findKnn(x, k, n, d);
    // Discard out of bounds neighbors : Not even a single one case
    if(d[0] > maxSquaredDistance){
        n.clear();
        d.clear();
        return;
    }
    // Discard out of bounds neighbors : General case
    for(IndexType i = 1 ; i < n.n_elem ; ++i){
        if(d[i] > maxSquaredDistance){
            n = n.rows(0, i-1);
            d = d.rows(0, i-1);
            break;
        }
    }
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>>
KDTree<IndexType, DecimalType>::findBoundedKnn(
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
void KDTree<IndexType, DecimalType>::findBoundedKnn(
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findBoundedKnn2D(
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
void KDTree<IndexType, DecimalType>::findBoundedKnn2D(
    PointType const &x,
    IndexType const k,
    DecimalType const maxSquaredDistance,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    // Obtain 2D knn
    findKnn2D(x, k, n, d);
    // Discard out of bounds neighbors : Not even a single one case
    if(d[0] > maxSquaredDistance){
        n.clear();
        d.clear();
        return;
    }
    // Discard out of bounds neighbors : General case
    for(IndexType i = 1 ; i < n.n_elem ; ++i){
        if(d[i] > maxSquaredDistance){
            n = n.rows(0, i-1);
            d = d.rows(0, i-1);
            break;
        }
    }
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>>
KDTree<IndexType, DecimalType>::findBoundedKnn2D(
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
void KDTree<IndexType, DecimalType>::findBoundedKnn2D(
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findSphere(
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
void KDTree<IndexType, DecimalType>::findSphere(
    PointType const &x,
    DecimalType const r,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    pcl::PointXYZ _x(x[0], x[1], x[2]);
    std::vector<int> _n;
    std::vector<float> _d;
    kdt3D->radiusSearch(_x, r, _n, _d);
    n = arma::conv_to<arma::Col<IndexType>>::from(_n);
    d = arma::conv_to<arma::Col<DecimalType>>::from(_d);
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>> KDTree<IndexType, DecimalType>::findSphere(
    arma::Mat<DecimalType> const &X,
    DecimalType const r
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    std::vector<arma::Col<DecimalType>> D(X.n_rows);
    findSphere(X, r, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void KDTree<IndexType, DecimalType>::findSphere(
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findCylinder(
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
void KDTree<IndexType, DecimalType>::findCylinder(
    PointType const &x,
    DecimalType const r,
    arma::Col<IndexType> &n,
    arma::Col<DecimalType> &d
){
    pcl::PointXY _x(x[0], x[1]);
    std::vector<int> _n;
    std::vector<float> _d;
    kdt2D->radiusSearch(_x, r, _n, _d);
    size_t const m = _n.size(); // Number of neighbors
    if(n.n_elem != m) n.resize(m); // Handle output indices size
    if(d.n_elem != m) d.resize(m); // Handle output distances size
    for(size_t i = 0 ; i < m ; ++i){ // Assign output indices and distances
        n[i] = _n[i];
        d[i] = _d[i];
    }
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>> KDTree<IndexType, DecimalType>::findCylinder(
    arma::Mat<DecimalType> const &X,
    DecimalType const r
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    std::vector<arma::Col<DecimalType>> D(X.n_rows);
    findCylinder(X, r, N, D);
    return N;
}

template <typename IndexType, typename DecimalType>
void KDTree<IndexType, DecimalType>::findCylinder(
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findBoundedCylinder(
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
void KDTree<IndexType, DecimalType>::findBoundedCylinder(
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
template <typename BoundaryCoordinateType>
std::vector<arma::Col<IndexType>>
KDTree<IndexType, DecimalType>::findBoundedCylinder(
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
template <typename BoundaryCoordinateType>
void KDTree<IndexType, DecimalType>::findBoundedCylinder(
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findBox(
    PointType const &x,
    arma::Col<DecimalType> const &halfLength
){
    arma::Col<IndexType> n;
    findBox(x, halfLength, n);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void KDTree<IndexType, DecimalType>::findBox(
    PointType const &x,
    arma::Col<DecimalType> const &halfLength,
    arma::Col<IndexType> &n
){
    // Determine radius of bounding sphere (min sphere containing the box)
    DecimalType const r = vl3dpp::math::SQRT_3 * arma::max(halfLength);
    // Extract bounding sphere
    arma::Col<DecimalType> _d;
    findSphere(x, r, n, _d);
    // Bounding box min vertex
    arma::Col<DecimalType> const a = x-halfLength;
    // Bounding box max vertex
    arma::Col<DecimalType> const b = x+halfLength;
    // Extract box-like neighborhood from bounding sphere
    arma::uvec to_remove(n.n_elem);
    IndexType next_remove_idx = 0;
    for(IndexType k = 0 ; k < n.n_elem ; ++k){ // For each k-th point in ngbhd.
        pcl::PointXYZ const &p = pcloud3D->at(n[k]);
        // Check if k-th point is out of the box
        if(
            p.x < a[0] || p.x > b[0] ||
            p.y < a[1] || p.y > b[1] ||
            p.z < a[2] || p.z > b[2]
        ) {
            // Flag k-th point to be removed from the neighborhood
            to_remove[next_remove_idx] = k;
            ++next_remove_idx;
        }
    }
    n.shed_rows(to_remove.head(next_remove_idx)); // Remove
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>> KDTree<IndexType, DecimalType>::findBox(
    arma::Mat<DecimalType> const &X,
    arma::Col<DecimalType> const &halfLength
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    findBox(X, halfLength, N);
    return N;
}

template <typename IndexType, typename DecimalType>
void KDTree<IndexType, DecimalType>::findBox(
    arma::Mat<DecimalType> const &X,
    arma::Col<DecimalType> const &halfLength,
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findRectangle(
    PointType const &x,
    arma::Col<DecimalType> const &halfLength
){
    arma::Col<IndexType> n;
    findRectangle(x, halfLength, n);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType>
void KDTree<IndexType, DecimalType>::findRectangle(
    PointType const &x,
    arma::Col<DecimalType> const &halfLength,
    arma::Col<IndexType> &n
){
    // Determine radius of bounding sphere (min sphere containing the box)
    DecimalType const r = vl3dpp::math::SQRT_2 * arma::max(halfLength);
    // Extract bounding sphere
    arma::Col<DecimalType> _d;
    findCylinder(x, r, n, _d);
    // Bounding box min vertex
    arma::Col<DecimalType> const a{x[0]-halfLength[0], x[1]-halfLength[1]};
    // Bounding box max vertex
    arma::Col<DecimalType> const b{x[0]+halfLength[0], x[1]+halfLength[1]};
    // Extract box-like neighborhood from bounding sphere
    arma::uvec to_remove(n.n_elem);
    IndexType next_remove_idx = 0;
    for(IndexType k = 0 ; k < n.n_elem ; ++k){ // For each k-th point in ngbhd.
        pcl::PointXY const &p = pcloud2D->at(n[k]);
        // Check if k-th point is out of the box
        if(
            p.x < a[0] || p.x > b[0] ||
            p.y < a[1] || p.y > b[1]
        ) {
            // Flag k-th point to be removed from the neighborhood
            to_remove[next_remove_idx] = k;
            ++next_remove_idx;
        }
    }
    n.shed_rows(to_remove.head(next_remove_idx)); // Remove
}

template <typename IndexType, typename DecimalType>
std::vector<arma::Col<IndexType>>
KDTree<IndexType, DecimalType>::findRectangle(
    arma::Mat<DecimalType> const &X,
    arma::Col<DecimalType> const &halfLength
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    findRectangle(X, halfLength, N);
    return N;
}

template <typename IndexType, typename DecimalType>
void KDTree<IndexType, DecimalType>::findRectangle(
    arma::Mat<DecimalType> const &X,
    arma::Col<DecimalType> const &halfLength,
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
arma::Col<IndexType> KDTree<IndexType, DecimalType>::findBoundedRectangle(
    PointType const &x,
    BoundaryCoordinateType const &z,
    arma::Col<DecimalType> const &halfLength,
    DecimalType const zmin,
    DecimalType const zmax
){
    arma::Col<IndexType> n;
    findBoundedRectangle(x, z, halfLength, zmin, zmax, n);
    return n;
}

template <typename IndexType, typename DecimalType>
template <typename PointType, typename BoundaryCoordinateType>
void KDTree<IndexType, DecimalType>::findBoundedRectangle(
    PointType const &x,
    BoundaryCoordinateType const &z,
    arma::Col<DecimalType> const &halfLength,
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
template <typename PointType, typename BoundaryCoordinateType>
std::vector<arma::Col<IndexType>>
KDTree<IndexType, DecimalType>::findBoundedRectangle(
    arma::Mat<DecimalType> const &X,
    BoundaryCoordinateType const &z,
    arma::Col<DecimalType> const &halfLength,
    DecimalType const zmin,
    DecimalType const zmax
){
    std::vector<arma::Col<IndexType>> N(X.n_rows);
    findBoundedRectangle(X, z, halfLength, zmin, zmax, N);
    return N;
}

template <typename IndexType, typename DecimalType>
template <typename PointType, typename BoundaryCoordinateType>
void KDTree<IndexType, DecimalType>::findBoundedRectangle(
    arma::Mat<DecimalType> const &X,
    BoundaryCoordinateType const &z,
    arma::Col<DecimalType> const &halfLength,
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

}