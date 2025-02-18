#include <alg/SupportNeighborhoods.hpp>

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType, typename LabelType, typename IndexType>
SupportNeighborhoods<XDecimalType, LabelType, IndexType>::SupportNeighborhoods(
    std::string const &nbhType,
    IndexType const nbhK,
    arma::Col<XDecimalType> const &nbhRadii,
    XDecimalType const nbhSeparationFactor,
    std::string const &strategy,
    IndexType const numPoints,
    short const fast,
    arma::Col<IndexType> const &trainingClassDistribution,
    bool const centerOnPcloud,
    bool const extraNodes,
    int const nthreads
) :
    nbhType(nbhType),
    nbhK(nbhK),
    nbhRadii(nbhRadii),
    nbhSeparationFactor(nbhSeparationFactor),
    strategy(strategy),
    numPoints(numPoints),
    fast(fast),
    trainingClassDistribution(trainingClassDistribution),
    centerOnPcloud(centerOnPcloud),
    extraNodes(extraNodes),
    nthreads(nthreads)
{
    // Handle as many threads as available
    if(this->nthreads==-1) this->nthreads=std::thread::hardware_concurrency();

    // Determine neighborhood method
    if(nbhType == "cylinder"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeCylindricalNeighborhood;
    }
    else if(nbhType == "bounded_cylinder"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeBoundedCylindricalNeighborhood;
    }
    else if(nbhType == "sphere"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeSphericalNeighborhood;
    }
    else if(nbhType == "rectangular2d"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeRectangular2DNeighborhood;
    }
    else if(nbhType == "rectangular3d"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeRectangular3DNeighborhood;
    }
    else if(nbhType == "knn2d"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeKnn2DNeighborhood;
    }
    else if(nbhType == "knn3d" || nbhType == "knn"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeKnn3DNeighborhood;
    }
    else if(nbhType == "bounded_knn2d"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeBoundedKnn2DNeighborhood;
    }
    else if(nbhType == "bounded_knn3d" || nbhType == "bounded_knn"){
        neighborhoodMethod =
            &SupportNeighborhoods::computeBoundedKnn3DNeighborhood;
    }
    else{
        std::stringstream ss;
        ss  << "SupportNeighborhoods cannot be instantiated with given "
            << "neighborhood type: \"" << nbhType << "\"";
        throw VL3DPPException(ss.str());
    }
}


// ***  MAIN METHODS  *** //
// ********************** //
template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<XDecimalType, LabelType, IndexType>::computeAll(
    arma::Mat<XDecimalType> const &X,
    arma::Col<LabelType> const &y,
    arma::Mat<XDecimalType> &Xsup,
    std::vector<arma::Col<IndexType>> &I
){
    // Compute the support points
    TimeWatcher twSup; twSup.start();
    computeSupportPoints(X, y, Xsup);
    twSup.stop();
    std::stringstream ss;
    ss  << "SupportNeighborhoods.computeAll computed " << Xsup.n_rows
        << " support points in "
        << twSup.getElapsedDecimalSeconds() << " seconds.";
    LOGGER->logInfo(ss.str());

    // Build KDTree to compute neighborhoods
    TimeWatcher twKDT; twKDT.start();
    KDTree<IndexType, XDecimalType> kdt = buildKDTree(X);
    twKDT.stop();
    ss.str("");
    ss  << "SupportNeighborhoods.computeAll built KDTree for "
        << X.n_rows << " points in "
        << twKDT.getElapsedDecimalSeconds() << " seconds.";
    LOGGER->logInfo(ss.str());

    // For each support point, compute its neighborhood
    TimeWatcher twNbh; twNbh.start();
    arma::uword const msup = Xsup.n_rows; // Number of support points
    I.resize(msup); // Allocate msup vectors of indices
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(X, Xsup, msup, I, kdt)
    for(arma::uword i = 0 ; i < msup ; ++i){ // Iterate supports with i
        arma::Col<XDecimalType> const &xsupi = Xsup.row(i).as_col(); // i-th pt
        (this->*neighborhoodMethod)(X, xsupi, kdt, I[i]); // i-th neighborhood
    }
    twNbh.stop();
    ss.str("");
    ss  << "SupportNeighborhoods.computeAll computed " << Xsup.n_rows
        << " support neighborhoods in "
        << twNbh.getElapsedDecimalSeconds() << " seconds.";
    LOGGER->logInfo(ss.str());
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<XDecimalType, LabelType, IndexType>::computeSupportPoints(
    arma::Mat<XDecimalType> const &X,
    arma::Col<LabelType> const &y,
    arma::Mat<XDecimalType> &Xsup
){
    if(strategy == "grid"){
        computeGridSupport(X, Xsup);
    }
    else if(strategy == "fps"){
        computeFPSSupport(X, Xsup);
    }
    else if(strategy == "training_class_distribution"){
        computeTrainingClassDistributionSupport(X, y, Xsup);
    }
    else{
        std::stringstream ss;
        ss  << "SupportNeighborhoods.computeSupportPoints failed due to an "
            << "unexpected strategy: \"" << strategy << "\"";
        throw VL3DPPException(ss.str());
    }
}


// ***  SUPPORT METHODS  *** //
// ************************* //
template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<XDecimalType, LabelType, IndexType>::computeGridSupport(
    arma::Mat<XDecimalType> const &X,
    arma::Mat<XDecimalType> &Xsup
){
    // Compute baseline support
    // TODO Rethink : Add knn 2D to 2D cases?
    if(nbhType == "cylinder" || nbhType == "rectangular2d"){
        arma::Col<XDecimalType> const cellSize = nbhRadii[0]*
            nbhSeparationFactor*
            arma::Col<XDecimalType>(2, arma::fill::ones);
        Xsup = GridMesher(cellSize, extraNodes).computeNodes(X.cols(0, 1));
        Xsup.insert_cols(
            2,
            arma::Col<XDecimalType>(Xsup.n_rows, arma::fill::zeros)
        );
    }
    else{
        arma::Col<XDecimalType> const cellSize = nbhRadii[0]*
            nbhSeparationFactor*
            arma::Col<XDecimalType>(3, arma::fill::ones);
        Xsup = GridMesher(cellSize, extraNodes).computeNodes(X);
    }
    // Center on point cloud (i.e., input structure space X), if requested
    if(centerOnPcloud) centerSupport(X, Xsup);

}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<XDecimalType, LabelType, IndexType>::computeFPSSupport(
    arma::Mat<XDecimalType> const &X,
    arma::Mat<XDecimalType> &Xsup
){
    // TODO Rethink : Remove bounded_knn2d from use2D cases?
    bool const use2D = nbhType == "cylinder" || nbhType == "rectangular2d" ||
        nbhType == "knn2d" || nbhType == "bounded_knn2d";
    if(fast==0){
        if(use2D){
            Xsup = FurthestPointSubsampler<XDecimalType>(
                numPoints, nthreads
            ).sample2D(X);
        }
        else{
            Xsup = FurthestPointSubsampler<XDecimalType>(
                numPoints, nthreads
            ).sample3D(X);
        }
    }
    else if(fast==1){
        if(use2D){
            Xsup = FurthestPointSubsampler<XDecimalType>(
                numPoints, nthreads
            ).hybridSample2D(X);
        }
        else{
            Xsup = FurthestPointSubsampler<XDecimalType>(
                numPoints, nthreads
            ).hybridSample3D(X);
        }
    }
    else if(fast==2){
        Xsup = FurthestPointSubsampler<XDecimalType>(
            numPoints, nthreads
        ).randomSample(X);
    }
    else if(fast==3){
        Xsup = FurthestPointSubsampler<XDecimalType>(
            numPoints, nthreads
        ).downSample(X);
    }
    else if(fast==4){
        if(use2D){
            Xsup = FurthestPointSubsampler<XDecimalType>(
                numPoints, nthreads
            ).parallelSample2D(X);
        }
        else{
            Xsup = FurthestPointSubsampler<XDecimalType>(
                numPoints, nthreads
            ).parallelSample3D(X);
        }
    }
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeTrainingClassDistributionSupport(
    arma::Mat<XDecimalType> const &X,
    arma::Col<LabelType> const &y,
    arma::Mat<XDecimalType> &Xsup
){
    // Prepare computation
    IndexType const nc = trainingClassDistribution.n_rows; // Num. classes
    IndexType const mSup = arma::sum(trainingClassDistribution);
    arma::Col<arma::uword> selectedIndices(mSup);
    // Select requested number of points per class
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE) \
        shared(nc, y, trainingClassDistribution, selectedIndices)
    for(IndexType c = 0 ; c < nc ; ++c){ // Iterate over classes
        arma::uvec Ic = arma::find(y==c); // Indices of points of class c
        Ic = arma::shuffle(Ic); // Randomly reordered indices
        // Find start output index for current class
        IndexType iStart = 0; // Offset to determine the start output index
        for(IndexType i = 0 ; i < c ; ++i){
            iStart += trainingClassDistribution[i];
        }
        // Store outputs in a concurrency-safe way
        for(IndexType i = 0 ; i < trainingClassDistribution[c] ; ++i){
            selectedIndices[i+iStart] = Ic[i];
        }
    }
    // Write selected points in support matrix
    Xsup = X.rows(selectedIndices);
}


// ***  NEIGHBORHOOD METHODS  *** //
// ****************************** //
template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeSphericalNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    arma::Col<XDecimalType> distances;
    kdt.findSphere(xsupi, nbhRadii[0], Ii, distances);
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeCylindricalNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    arma::Col<XDecimalType> distances;
    kdt.findCylinder(xsupi, nbhRadii[0], Ii, distances);
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeBoundedCylindricalNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    arma::Col<XDecimalType> distances;
    arma::Col<XDecimalType> const &z = X.col(2);
    kdt.findBoundedCylinder(
        xsupi, z, nbhRadii[0], nbhRadii[1], nbhRadii[2], Ii, distances
    );
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeRectangular2DNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    kdt.findRectangle(xsupi, nbhRadii, Ii);
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeRectangular3DNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    arma::Col<XDecimalType> const &z = X.col(2);
    kdt.findBoundedRectangle(
        xsupi, z, nbhRadii.subvec(0, 1), -nbhRadii[2], nbhRadii[2], Ii
    );
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeKnn2DNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    arma::Col<XDecimalType> d;
    kdt.findKnn2D(xsupi, nbhK, Ii, d);
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeKnn3DNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    arma::Col<XDecimalType> d;
    kdt.findKnn(xsupi, nbhK, Ii, d);
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeBoundedKnn2DNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    arma::Col<XDecimalType> d;
    kdt.findBoundedKnn2D(xsupi, nbhK, nbhRadii[0], Ii, d);
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<
    XDecimalType, LabelType, IndexType
>::computeBoundedKnn3DNeighborhood(
    arma::Mat<XDecimalType> const &X,
    arma::Col<XDecimalType> const &xsupi,
    KDTree<IndexType, XDecimalType> &kdt,
    arma::Col<IndexType> &Ii
){
    arma::Col<XDecimalType> d;
    kdt.findBoundedKnn(xsupi, nbhK, nbhRadii[0], Ii, d);
}


// ***   UTIL METHODS   *** //
// ************************ //
template <typename XDecimalType, typename LabelType, typename IndexType>
KDTree<IndexType, XDecimalType>
SupportNeighborhoods<XDecimalType, LabelType, IndexType>::buildKDTree(
    arma::Mat<XDecimalType> const &X
){
    // Determine neighborhood method
    if(nbhType == "cylinder"){
        return KDTree<IndexType, XDecimalType>(X, true, false);
    }
    else if(nbhType == "bounded_cylinder"){
        return KDTree<IndexType, XDecimalType>(X, true, false);
    }
    else if(nbhType == "sphere"){
        return KDTree<IndexType, XDecimalType>(X, false, true);
    }
    else if(nbhType == "rectangular2d"){
        return KDTree<IndexType, XDecimalType>(X, true, false);
    }
    else if(nbhType == "rectangular3d"){
        return KDTree<IndexType, XDecimalType>(X, true, false);
    }
    else if(nbhType == "knn2d"){
        return KDTree<IndexType, XDecimalType>(X, true, false);
    }
    else if(nbhType == "knn3d" || nbhType == "knn"){
        return KDTree<IndexType, XDecimalType>(X, false, true);
    }
    else if(nbhType == "bounded_knn2d"){
        return KDTree<IndexType, XDecimalType>(X, true, false);
    }
    else if(nbhType == "bounded_knn3d" || nbhType == "bounded_knn"){
        return KDTree<IndexType, XDecimalType>(X, false, true);
    }
    else{
        std::stringstream ss;
        ss  << "SupportNeighborhoods cannot build KDTree for requested "
            << "neighborhood type: \"" << nbhType << "\"";
        throw VL3DPPException(ss.str());
    }
}

template <typename XDecimalType, typename LabelType, typename IndexType>
void
SupportNeighborhoods<XDecimalType, LabelType, IndexType>::centerSupport(
    arma::Mat<XDecimalType> const &X,
    arma::Mat<XDecimalType> &Xsup
){
    // Build KDTree for 3D bounded KNN
    bool const use2D =  nbhType=="cylinder" || nbhType=="bounded_cylinder" ||
        nbhType=="rectangular2d" ||
        nbhType=="knn2d" || nbhType=="bounded_knn2d";
    KDTree<IndexType, XDecimalType> kdt(X, use2D, !use2D);
    void(KDTree<IndexType, XDecimalType>::*kdtQuery)(
        arma::Col<XDecimalType> const &,
        IndexType const,
        arma::Col<IndexType> &,
        arma::Col<XDecimalType> &
    ) = use2D ?
        (
            void(KDTree<IndexType, XDecimalType>::*)(
                arma::Col<XDecimalType> const &,
                IndexType const,
                arma::Col<IndexType> &,
                arma::Col<XDecimalType> &
            )
        )&KDTree<IndexType, XDecimalType>::findKnn2D :
        (
            void(KDTree<IndexType, XDecimalType>::*)(
                arma::Col<XDecimalType> const &,
                IndexType const,
                arma::Col<IndexType> &,
                arma::Col<XDecimalType> &
            )
        )&KDTree<IndexType, XDecimalType>::findKnn;

    // Make sure all support points come from the input point cloud
    arma::uword const m = X.n_rows; // Number of points (in point cloud)
    arma::uword const mSup = Xsup.n_rows; // Number of support points
    XDecimalType const squaredDistanceThreshold = nbhRadii[0]*nbhRadii[0];
    arma::Col<arma::u8> selMask(m, arma::fill::zeros); // Selection mask
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(m, mSup, Xsup, kdt, kdtQuery, selMask, squaredDistanceThreshold)
    for(arma::uword i = 0 ; i < mSup ; ++i){
        arma::Col<XDecimalType> const &xsupi = Xsup.row(i).as_col();
        arma::Col<IndexType> I;
        arma::Col<XDecimalType> d;
        (kdt.*kdtQuery)(xsupi, 1, I, d);
        if(d[0] <= squaredDistanceThreshold){ // Replace by closest neighbor
            selMask[I[0]] = 1;
        }
    }
    Xsup = X.rows(arma::find(selMask));
}

