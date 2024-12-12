#include <mining/SmoothFeatsMiner.hpp>

using vl3dpp::mining::SmoothFeatsMiner;
using vl3dpp::adt::kdtree::KDTree;
using vl3dpp::adt::octree::Octree;
using vl3dpp::util::TimeWatcher;
using vl3dpp::util::VL3DPPException;


// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType, typename FDecimalType>
SmoothFeatsMiner<XDecimalType, FDecimalType>::SmoothFeatsMiner(
    string const &nbhType,
    int const nbhK,
    arma::Col<XDecimalType> const nbhRadius,
    XDecimalType const nbhLowerBound,
    XDecimalType const nbhUpperBound,
    FDecimalType const weightedMeanOmega,
    FDecimalType const gaussianRbfOmega,
    string const &nanPolicy,
    vector<std::string> &fnames,
    int const nthreads
) :
    nbhType(nbhType),
    nbhK(nbhK),
    nbhRadius(nbhRadius),
    nbhLowerBound(nbhLowerBound),
    nbhUpperBound(nbhUpperBound),
    weightedMeanOmega(weightedMeanOmega),
    gaussianRbfOmega(gaussianRbfOmega),
    nanPolicy(nanPolicy),
    fnames(fnames),
    nthreads(nthreads),
    squaredGaussianRbfOmega(gaussianRbfOmega*gaussianRbfOmega)
{
    // Handle number of threads -1
    if(this->nthreads==-1) this->nthreads=std::thread::hardware_concurrency();
    // Assign NaN policy method
    if(nanPolicy == "propagate") nanHandlingFunction = nanPropagate;
    else if(nanPolicy == "replace") nanHandlingFunction = nanReplace;
    else {
        std::stringstream ss;
        ss  << "SmoothFeatsMinerPP received an unexpected NaN policy: "
            << "\"" << nanPolicy << "\"";
        throw VL3DPPException(ss.str());
    }
    // Assign smooth features computation functions
    for(string const &fname : fnames){
        if(fname == "mean"){
            smoothFeaturesFunctions.push_back(computeMeanSmooth);
            smoothFeaturesOmegas.push_back(0);
        }
        else if(fname == "weighted_mean"){
            smoothFeaturesFunctions.push_back(computeWeightedMeanSmooth);
            smoothFeaturesOmegas.push_back(weightedMeanOmega);
        }
        else if(fname == "gaussian_rbf"){
            smoothFeaturesFunctions.push_back(computeGaussianRbfSmooth);
            smoothFeaturesOmegas.push_back(squaredGaussianRbfOmega);
        }
        else{
            throw VL3DPPException(
                "SmoothFeatsMiner failed to recognize feature name \"" +
                fname + "\"."
            );
        }
    }
};


// ***  DATA MINING METHODS  *** //
// ***************************** //
template <typename XDecimalType, typename FDecimalType>
arma::Mat<FDecimalType> SmoothFeatsMiner<XDecimalType, FDecimalType>::mine(
    arma::Mat<XDecimalType> const &X,
    arma::Mat<FDecimalType> const &F
){
    // Start global time measurement
    TimeWatcher twGlobal; twGlobal.start();

    // Build neighborhood engine
    TimeWatcher twNbhEngine; twNbhEngine.start();
    NeighborhoodEngine const nbhEngine = buildNeighborhoodEngine(X);
    twNbhEngine.stop();
    std::stringstream ss;
    ss  << "SmoothFeatsMinerPP built the neighborhood engine for " << X.n_rows
        << " points in " << twNbhEngine.getElapsedDecimalSeconds()
        << " seconds.";
    LOGGER->logInfo(ss.str());

    // Prepare smooth features computation
    TimeWatcher twComp; twComp.start();
    arma::uword const nf = F.n_cols * // Number of features
        smoothFeaturesFunctions.size();
    arma::uword const m = X.n_rows; // Number of points
    arma::subview_col<XDecimalType> const &z = X.col(2); // For bounded ngbhds.
    arma::Mat<FDecimalType> Fhat(m, nf); // Matrix of smooth features
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most

    // Compute point-wise smooth features
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(X, z, F, Fhat, nbhEngine, m)
    for(arma::uword i = 0 ; i < m ; ++i){ // Smooth features for each point
        computeSmoothFeatures(X.row(i), i, X, z, F, nbhEngine, Fhat.row(i));
    }

    // Report smooth features computation
    twComp.stop();
    ss.str("");
    ss  << "SmoothFeatsMinerPP computed the " << m << " x " << nf << " matrix "
        << "of smooth features in " << twComp.getElapsedDecimalSeconds()
        << " seconds.";
    LOGGER->logInfo(ss.str());

    // Report total time
    twGlobal.stop();
    ss.str("");
    ss  << "SmoothFeatsMinerPP finished in "
        << twGlobal.getElapsedDecimalSeconds()
        << " seconds.";
    LOGGER->logInfo(ss.str());

    // Return
    return Fhat;
}

template <typename XDecimalType, typename FDecimalType>
void SmoothFeatsMiner<XDecimalType, FDecimalType>::computeSmoothFeatures(
    arma::subview_row<XDecimalType> const &xi,
    arma::uword const i,
    arma::Mat<XDecimalType> const &X,
    arma::subview_col<XDecimalType> const &z,
    arma::Mat<FDecimalType> const &F,
    NeighborhoodEngine const &nbhEngine,
    arma::subview_row<FDecimalType> fout
) const {
    // Find the neighborhood
    arma::Col<arma::uword> const N = nbhEngine.query(
        xi.as_col(),
        z,
        nbhEngine.queryer,
        nbhK,
        nbhRadius,
        nbhLowerBound,
        nbhUpperBound
    );
    // Extract structure and features for the neighborhood
    arma::Mat<XDecimalType> XN(N.n_rows, X.n_cols);
    arma::Mat<FDecimalType> FN(N.n_rows, F.n_cols);
    for(arma::uword k = 0 ; k < N.n_rows ; ++k){
        arma::uword const j = N[k];
        XN.row(k) = X.row(j);
        FN.row(k) = F.row(j);
    }
    // Handle NaN values
    nanHandlingFunction(FN);
    // Compute the smooth features themselves
    arma::uword nextFeatIdx = 0;
    for(size_t k = 0 ; k < smoothFeaturesFunctions.size() ; ++k){
        smoothFeaturesFunctions[k](
            xi,
            XN,
            FN,
            smoothFeaturesOmegas[k],
            nextFeatIdx,
            fout
        );
    }
}


// ***  NEIGHBORHOOD ENGINE METHODS  *** //
// ************************************* //
template <typename XDecimalType, typename FDecimalType>
typename SmoothFeatsMiner<XDecimalType, FDecimalType>::NeighborhoodEngine
SmoothFeatsMiner<XDecimalType, FDecimalType>::buildNeighborhoodEngine(
    arma::Mat<XDecimalType> const &X
){
    NeighborhoodEngine ne;
    // K-nearest neighborhoods (3D)
    if(nbhType == "knn"){
        ne.queryer.kdt = new KDTree<arma::uword, XDecimalType>(
            X, false, true
        );
        ne.query = [](
            arma::Col<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            typename NeighborhoodEngine::Queryer const &queryer,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return queryer.kdt->findKnn(xi, k);
        };
    }
    // K-nearest neighborhoods (2D)
    else if(nbhType == "knn2d"){
        ne.queryer.kdt = new KDTree<arma::uword, XDecimalType>(
            X, true, false
        );
        ne.query = [](
            arma::Col<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            typename NeighborhoodEngine::Queryer const &queryer,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return queryer.kdt->findKnn2D(xi, k);
        };
    }
    // Spherical neighborhoods
    else if(nbhType == "sphere"){
        ne.queryer.kdt = new KDTree<arma::uword, XDecimalType>(
            X, false, true
        );
        ne.query = [](
            arma::Col<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            typename NeighborhoodEngine::Queryer const &queryer,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return queryer.kdt->findSphere(xi, radius[0]);
        };
    }
    // Cylindrical neighborhoods
    else if(nbhType == "cylinder"){
        ne.queryer.kdt = new KDTree<arma::uword, XDecimalType>(
            X, true, false
        );
        ne.query = [](
            arma::Col<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            typename NeighborhoodEngine::Queryer const &queryer,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return queryer.kdt->findCylinder(xi, radius[0]);
        };
    }
    // Rectangular 3D neighborhoods
    else if(nbhType == "rectangular3d"){
        ne.queryer.kdt = new KDTree<arma::uword, XDecimalType>(
            X, false, true
        );
        ne.query = [](
            arma::Col<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            typename NeighborhoodEngine::Queryer const &queryer,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return queryer.kdt->findBox(
                xi,
                radius
            );
        };
    }
    // Rectangular 2D neighborhoods
    else if(nbhType == "rectangular2d"){
        ne.queryer.kdt = new KDTree<arma::uword, XDecimalType>(
            X, true, false
        );
        ne.query = [](
            arma::Col<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            typename NeighborhoodEngine::Queryer const &queryer,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return queryer.kdt->findRectangle(xi, radius);
        };
    }
    // Bounded cylinder
    else if(nbhType == "boundedcylinder"){
        ne.queryer.kdt = new KDTree<arma::uword, XDecimalType>(
            X, true, false
        );
        ne.query = [](
            arma::Col<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            typename NeighborhoodEngine::Queryer const &queryer,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return queryer.kdt->findBoundedCylinder(
                xi,
                z,
                radius[0],
                lowerBound,
                upperBound
            );
        };
    }
    // Unexpected neighborhood type
    else{
        std::stringstream ss;
        ss  << "SmoothFeatsMinerPP received an unexpected neighborhood type "
            << "\"" << nbhType << "\"";
        throw VL3DPPException(ss.str());
    }
    // Return built neighborhood engine
    return ne;
}


// ***  NAN POLICY METHODS  *** //
// **************************** //
template <typename XDecimalType, typename FDecimalType>
void SmoothFeatsMiner<XDecimalType, FDecimalType>::nanPropagate(
    arma::Mat<FDecimalType> &FN
){
    return;
}

template <typename XDecimalType, typename FDecimalType>
void SmoothFeatsMiner<XDecimalType, FDecimalType>::nanReplace(
    arma::Mat<FDecimalType> &FN
){
    for(arma::uword j = 0 ; j < FN.n_cols ; ++j){
        arma::subview_col<FDecimalType> fj = FN.col(j);
        arma::Col<arma::uword> const Inan = arma::find_nan(fj);
        arma::Col<arma::uword> const InotNan = arma::find_finite(fj);
        if(InotNan.n_elem < 1) continue; // Nothing to do if all values are nan
        fj.each_row(Inan) = arma::mean(
            FN.submat(InotNan, arma::Col<arma::uword>({j}))
        );
    }
}


// ***  SMOOTH FEATURES METHODS  *** //
// ********************************* //
template <typename XDecimalType, typename FDecimalType>
void SmoothFeatsMiner<XDecimalType, FDecimalType>::computeMeanSmooth(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    FDecimalType const omega,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> fout
){
    fout.cols(nextFeatIdx, nextFeatIdx+FN.n_cols-1) = arma::mean(FN, 0);
    nextFeatIdx += FN.n_cols;
}

template <typename XDecimalType, typename FDecimalType>
void SmoothFeatsMiner<XDecimalType, FDecimalType>::computeWeightedMeanSmooth(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    FDecimalType const omega,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> fout
){
    // Compute point-wise distance weights (d), max (dmax), and denominator (D)
    arma::Col<FDecimalType> const norm = arma::conv_to<
        arma::Col<FDecimalType>
    >::from(arma::sqrt(
        arma::sum(arma::square(XN.each_row()-xi), 1)
    ));
    FDecimalType const dmax = arma::max(norm);
    arma::Col<FDecimalType> const d = dmax - norm + omega;
    FDecimalType const D = arma::sum(d);
    // Compute the weighted mean
    for(arma::uword j = 0 ; j < FN.n_cols ; ++j, ++nextFeatIdx){
        fout.col(nextFeatIdx) = arma::sum(FN.col(j)%d)/D;
    }
}

template <typename XDecimalType, typename FDecimalType>
void SmoothFeatsMiner<XDecimalType, FDecimalType>::computeGaussianRbfSmooth(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    FDecimalType const squaredOmega,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> fout
){
    // Compute the distance weights (d) and denominator (D)
    arma::Col<FDecimalType> const d = arma::conv_to<
        arma::Col<FDecimalType>
    >::from(arma::exp(
        -arma::sum(arma::square(XN.each_row()-xi), 1)/squaredOmega
    ));
    FDecimalType const D = arma::sum(d);
    // Compute the Gaussian RBF
    for(arma::uword j = 0 ; j < FN.n_cols ; ++j, ++nextFeatIdx){
        fout.col(nextFeatIdx) = arma::sum(FN.col(j)%d)/D;
    }
}
