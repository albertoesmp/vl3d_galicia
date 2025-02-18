#include <mining/HeightFeatsMiner.hpp>

using vl3dpp::mining::HeightFeatsMiner;
using vl3dpp::adt::kdtree::KDTree;
using vl3dpp::util::TimeWatcher;
using vl3dpp::util::VL3DPPException;


// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType, typename FDecimalType>
HeightFeatsMiner<XDecimalType, FDecimalType>::HeightFeatsMiner(
    string const &nbhType,
    XDecimalType const nbhRadius,
    XDecimalType const nbhSeparationFactor,
    string const &outlierFilter,
    vector<string> const &fnames,
    int const nthreads
) :
    nbhType(nbhType),
    nbhRadius(nbhRadius),
    nbhSeparationFactor(nbhSeparationFactor),
    outlierFilter(outlierFilter),
    fnames(fnames),
    numFeatures(0),
    nthreads(nthreads)
{
    // Handle number of threads -1
    if(this->nthreads==-1) this->nthreads=std::thread::hardware_concurrency();
    // Assign corresponding outlier filtering function
    std::transform( // Outlier filter string to lower case
        this->outlierFilter.begin(),
        this->outlierFilter.end(),
        this->outlierFilter.begin(),
        [](unsigned char c){return std::tolower(c);}
    );
    if(this->outlierFilter == "iqr"){
        outlierFilterFunction = this->filterByIQR;
    }
    else if(this->outlierFilter == "stdev"){
        outlierFilterFunction = this->filterByStdev;
    }
    else if(this->outlierFilter == ""){
        outlierFilterFunction = [](arma::Col<FDecimalType> &z){return;};
    }
    else{
        throw VL3DPPException(
            "HeightFeatsMinerPP failed to recognize outlier filter \"" +
            outlierFilter + "\"."
        );
    }
    // Assign corresponding neighborhood finding function
    if(this->nbhType == "cylinder"){
        findNeighborhoodFunction = [] (
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::Row<XDecimalType> const &xi,
            XDecimalType const r
        ) -> arma::Col<arma::uword> {
            return kdt.findCylinder(xi, r);
        };
    }
    else if(this->nbhType == "rectangular2d"){
        findNeighborhoodFunction = [] (
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::Row<XDecimalType> const &xi,
            XDecimalType const r
        ) -> arma::Col<arma::uword> {
            return kdt.template findRectangle<arma::Col<XDecimalType>>(
                xi.as_col(),
                arma::Col<XDecimalType>{r, r}
            );
        };
    }
    else{
        throw VL3DPPException(
            "HeightFeatsMinerPP failed to recognize neighborhood type \"" +
            nbhType + "\"."
        );
    }
    // Assign height features computation functions
    for(string const &fname : fnames){
        if(fname == "floor_distance"){
            heightFeaturesFunctions.push_back(computeFloorDistance);
            ++numFeatures;
        }
        else if(fname == "ceil_distance"){
            heightFeaturesFunctions.push_back(computeCeilDistance);
            ++numFeatures;
        }
        else if(fname == "floor_coordinate"){
            heightFeaturesFunctions.push_back(computeFloorCoordinate);
            ++numFeatures;
        }
        else if(fname == "ceil_coordinate"){
            heightFeaturesFunctions.push_back(computeCeilCoordinate);
            ++numFeatures;
        }
        else if(fname == "height_range"){
            heightFeaturesFunctions.push_back(computeHeightRange);
            ++numFeatures;
        }
        else if(fname == "mean_height"){
            heightFeaturesFunctions.push_back(computeMeanHeight);
            ++numFeatures;
        }
        else if(fname == "median_height"){
            heightFeaturesFunctions.push_back(computeMedianHeight);
            ++numFeatures;
        }
        else if(fname == "height_quartiles"){
            heightFeaturesFunctions.push_back(computeHeightQuartiles);
            numFeatures += 3;
        }
        else if(fname == "height_deciles"){
            heightFeaturesFunctions.push_back(computeHeightDeciles);
            numFeatures += 9;
        }
        else if(fname == "height_variance"){
            heightFeaturesFunctions.push_back(computeHeightVariance);
            ++numFeatures;
        }
        else if(fname == "height_stdev"){
            heightFeaturesFunctions.push_back(computeHeightStdev);
            ++numFeatures;
        }
        else if(fname == "height_skewness"){
            heightFeaturesFunctions.push_back(computeHeightSkewness);
            ++numFeatures;
        }
        else if(fname == "height_kurtosis"){
            heightFeaturesFunctions.push_back(computeHeightKurtosis);
            ++numFeatures;
        }
        else{
            throw VL3DPPException(
                "HeightFeatsMiner failed to recognize feature name \"" +
                fname + "\"."
            );
        }
    }
}

// ***  DATA MINING METHODS  *** //
// ***************************** //
template <typename XDecimalType, typename FDecimalType>
arma::Mat<FDecimalType> HeightFeatsMiner<
    XDecimalType, FDecimalType
>::mine(
    arma::Mat<XDecimalType> const &X
) {
    // Start global time measurement
    TimeWatcher twGlobal; twGlobal.start();

    // Build 2D KDTree
    TimeWatcher twKDT; twKDT.start();
    KDTree<arma::uword, XDecimalType> kdt(X, true, false);
    twKDT.stop();
    std::stringstream ss;
    ss  << "HeightFeatsMinerPP built the KDTree for " << X.n_rows << " points "
        << "in " << twKDT.getElapsedDecimalSeconds() << " seconds.";
    LOGGER->logInfo(ss.str());

    // Prepare height features computation
    TimeWatcher twComp; twComp.start();
    arma::uword const m = X.n_rows; // Number of points
    arma::Mat<FDecimalType> F(m, numFeatures); // Matrix of height features
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most

    // Compute point-wise height features
    if(nbhSeparationFactor == 0.0) {
        #pragma omp parallel for default(none) \
            schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
            shared(X, F, kdt, m)
        for (arma::uword i = 0; i < m; ++i) { // Height features for each point
            computeHeightFeatures(X.row(i), i, X, kdt, F);
        }
    }

    // Compute support height features and their point-wise propagations
    else if(nbhSeparationFactor > 0.0){
        // Build grid
        TimeWatcher twGrid; twGrid.start();
        XDecimalType const halfLength = nbhSeparationFactor * nbhRadius;
        LazySupportGrid<XDecimalType> supGrid(
            arma::Col<XDecimalType>({halfLength, halfLength}),
            X.cols(0, 1)
        );
        size_t const supm = supGrid.getNumCells();
        twGrid.stop();
        ss.str("");
        ss  << "HeightFeatsMinerPP built support grid with " << supm
            << " cells in " << twGrid.getElapsedDecimalSeconds()
            << " seconds.";
        LOGGER->logInfo(ss.str());
        // Compute features for the support grid
        TimeWatcher twSupF; twSupF.start();
        // Support ceil and floor distances must consider only the coordinates
        std::vector<HeightFeatureFunction> const heightFeaturesFunctions =
            this->heightFeaturesFunctions;
        for(size_t i = 0 ; i < fnames.size() ; ++i){
            std::string const fname = fnames[i];
            if(fname == "floor_distance"){
                this->heightFeaturesFunctions[i] = computeFloorCoordinate;
            }
            else if(fname == "ceil_distance"){
                this->heightFeaturesFunctions[i] = computeCeilCoordinate;
            }
        }
        // The computation of the support features themselves
        arma::Mat<FDecimalType> supF(supm, numFeatures); // Support features
        #pragma omp parallel for default(none) \
            schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
            shared(supGrid, X, kdt, supF, supm, m)
        for(arma::uword i = 0 ; i < supm ; ++i){ // Height features on support
            computeHeightFeatures(
                supGrid.getCentroid(i).as_row(),
                i,
                X,
                kdt,
                supF
            );
        }
        twSupF.stop();
        ss.str("");
        ss  << "HeightFeatsMinerPP computed " << supm << " support features "
            << "in " << twSupF.getElapsedDecimalSeconds()
            << " seconds.";
        LOGGER->logInfo(ss.str());
        // Propagate support features to point cloud
        TimeWatcher twProp; twProp.start();
        #pragma omp parallel for default(none) \
            schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
            shared(supGrid, X, supF, F, m)
        for(arma::uword i = 0 ; i < m ; ++i){ // Propagate height features
            size_t const cellIdx = supGrid.findCellIndex(
                X.row(i).cols(0, 1).as_col()
            );
            F.row(i) = supF.row(cellIdx);
        }
        // Compute final vertical distances from propagated support features
        arma::Col<FDecimalType> const z = arma::conv_to<
            arma::Col<FDecimalType>
        >::from(X.col(2));
        for(arma::uword i=0, j=0 ; i < fnames.size() ; ++i, ++j){
            if(fnames[i]=="floor_distance"){
                F.col(j) = z-F.col(j);
            }
            else if(fnames[i]=="ceil_distance"){
                F.col(j) = F.col(j)-z;
            }
            else if(fnames[i]=="height_quartiles"){
                j += 2;
            }
            else if(fnames[i]=="height_deciles"){
                j += 8;
            }
        }
        // Restore original member attributes
        this->heightFeaturesFunctions = heightFeaturesFunctions;
        twProp.stop();
        ss.str("");
        ss  << "HeightFeatsMinerPP propagated support height features in "
            << twProp.getElapsedDecimalSeconds()
            << " seconds.";
        LOGGER->logInfo(ss.str());
    }

    // Unexpected separation factor
    else{
        ss.str("");
        ss  <<  "HeightFeatsMinerPP failed to compute height features due to "
                "an unexpected neighborhood separation factor ("
                << nbhSeparationFactor << ")";
        throw VL3DPPException(ss.str());
    }

    // Report height features computation
    twComp.stop();
    ss.str("");
    ss << "HeightFeatsMinerPP computed the " << m << " x " << numFeatures
       << " matrix of height features in " << twComp.getElapsedDecimalSeconds()
       << " seconds.";
    LOGGER->logInfo(ss.str());

    // Report total time
    twGlobal.stop();
    ss.str("");
    ss  << "HeightFeatsMinerPP finished in "
        << twGlobal.getElapsedDecimalSeconds()
        << " seconds.";
    LOGGER->logInfo(ss.str());

    // Return
    return F;
}

template <typename XDecimalType, typename FDecimalType>
void HeightFeatsMiner<
    XDecimalType, FDecimalType
>::computeHeightFeatures(
    arma::Row<XDecimalType> const &x,
    arma::uword const i,
    arma::Mat<XDecimalType> const &X,
    KDTree<arma::uword, XDecimalType> &kdt,
    arma::Mat<FDecimalType> &F
) const {
    // Find the neighborhood
    arma::Col<arma::uword> const N = findNeighborhoodFunction(
        kdt, x, nbhRadius
    );
    // Handle empty neighborhoods
    if(N.n_elem < 1){
        F.row(i).fill(arma::datum::nan);
        return;
    }
    // Extract vertical values from the neighborhood
    arma::Col<FDecimalType> z(N.n_elem);
    for(arma::uword j = 0 ; j < N.n_elem ; ++j) z[j] = X.at(N[j], 2);
    outlierFilterFunction(z);
    // Compute the height features themselves
    bool minComputed=false, maxComputed=false, meanComputed = false;
    bool stdevComputed = false;
    FDecimalType min, max, mean, stdev;
    arma::uword nextFeatIdx = 0;
    arma::subview_row<FDecimalType> fi = F.row(i);
    size_t const numFeatureNames = fnames.size();
    for(size_t k = 0 ; k < numFeatureNames ; ++k){
        heightFeaturesFunctions[k](
            x,
            z,
            minComputed,
            maxComputed,
            meanComputed,
            stdevComputed,
            min,
            max,
            mean,
            stdev,
            nextFeatIdx,
            fi
        );
    }
}


// ***  OUTLIER FILTERING METHODS  *** //
// *********************************** //
template <typename XDecimalType, typename FDecimalType>
void HeightFeatsMiner<XDecimalType, FDecimalType>::filterByIQR(
    arma::Col<FDecimalType> &z
) {
    // Nothing to filter if there are three or fewer points
    if(z.n_elem <= 3) return;
    // Compute the quartiles
    arma::Col<FDecimalType> Q = arma::quantile(
        z, arma::Col<FDecimalType>{0.25, 0.50, 0.75}
    );
    // Compute admissible interval
    FDecimalType const IQR = Q[2]-Q[0];
    FDecimalType const a = Q[0]-IQR;
    FDecimalType const b = Q[2]+IQR;
    // Filter out points outside the admissible interval
    z.shed_rows((z<a) && (z>b));
}

template <typename XDecimalType, typename FDecimalType>
void HeightFeatsMiner<XDecimalType, FDecimalType>::filterByStdev(
    arma::Col<FDecimalType> &z
) {
    // Nothing to filter if there are three or fewer points
    if(z.n_elem <= 3) return;
    // Compute the mean and standard deviation
    FDecimalType const mu = arma::mean(z);
    FDecimalType const sigma = arma::stddev(z);
    // Compute admissible interval
    FDecimalType const a = mu-3.0*sigma;
    FDecimalType const b = mu+3.0*sigma;
    // Filter out points outside the admissible interval
    z.shed_rows((z<a) && (z>b));
}


// ***  HEIGHT FEATURES METHODS  *** //
// ********************************* //
template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeFloorCoordinate(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Return if already computed
    if(minComputed){
        out[nextFeatIdx] = min;
        ++nextFeatIdx;
        return;
    }
    // Otherwise, comptue and return
    minComputed = true;
    min = arma::min(z);
    out[nextFeatIdx] = min;
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeCeilCoordinate(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Return if already computed
    if(maxComputed){
        out[nextFeatIdx] = max;
        ++nextFeatIdx;
        return;
    }
    // Otherwise, compute and return
    maxComputed = true;
    max = arma::max(z);
    out[nextFeatIdx] = max;
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeFloorDistance(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Compute min z if it has not been computed yet
    if(!minComputed){
        min = arma::min(z);
        minComputed = true;
    }
    // Return floor distance
    out[nextFeatIdx] = xi[2]-min;
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeCeilDistance(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Compute max z if it has not been computed yet
    if(!maxComputed){
        max = arma::max(z);
        maxComputed = true;
    }
    // Return floor distance
    out[nextFeatIdx] = max-xi[2];
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeHeightRange(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Compute max z if it has not been computed yet
    if(!maxComputed){
        max = arma::max(z);
        maxComputed = true;
    }
    // Compute min z if it has not been computed yet
    if(!minComputed){
        min = arma::max(z);
        minComputed = true;
    }
    // Compute the range
    out[nextFeatIdx] = max-min;
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeMeanHeight(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // If mean was already calculated, return it
    if(meanComputed) {
        out[nextFeatIdx] = mean;
        ++nextFeatIdx;
        return;
    }
    // Compute the mean
    mean = arma::mean(z);
    meanComputed = true;
    out[nextFeatIdx] = mean;
    ++nextFeatIdx;
}


template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeMedianHeight(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Compute the median
    out[nextFeatIdx] = arma::median(z);
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeHeightQuartiles(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Compute the quartiles
    arma::Col<FDecimalType> const Q = arma::quantile(
        z,
        arma::Col<FDecimalType>({0.25, 0.5, 0.75})
    );
    for(size_t i = 0 ; i < 3 ; ++i, ++nextFeatIdx) out[nextFeatIdx] = Q[i];
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeHeightDeciles(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Compute the quartiles
    arma::Col<FDecimalType> const Q = arma::quantile(
        z,
        arma::Col<FDecimalType>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9})
    );
    for(size_t i = 0 ; i < 9 ; ++i, ++nextFeatIdx) out[nextFeatIdx] = Q[i];
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeHeightVariance(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // If stdev is computed, derive variance from it
    if(stdevComputed){
        out[nextFeatIdx] = stdev*stdev;
        ++nextFeatIdx;
        return;
    }
    // Compute the standard deviation
    stdev = arma::stddev(z);
    stdevComputed = true;
    // Compute the variance from the standard deviation
    out[nextFeatIdx] = stdev*stdev;
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeHeightStdev(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // If stdev is computed, return it
    if(stdevComputed){
        out[nextFeatIdx] = stdev;
        ++nextFeatIdx;
        return;
    }
    // Compute the standard deviation
    stdev = arma::stddev(z);
    stdevComputed = true;
    out[nextFeatIdx] = stdev;
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeHeightSkewness(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Compute mean if it has not been computed before
    if(!meanComputed){
        mean = arma::mean(z);
        meanComputed = true;
    }
    // Compute stdev if it has not been computed before
    if(!stdevComputed){
        stdev = arma::stddev(z);
        stdevComputed = true;
    }
    // Compute the skewness
    if(stdev < 1e-6){
        out[nextFeatIdx] = 0.0;
    }
    else{
        arma::Col<FDecimalType> const y =  (z-mean)/stdev;
        out[nextFeatIdx] = arma::mean(y%y%y%y);
    }
    ++nextFeatIdx;
}

template <typename XDecimalType, typename FDecimalType>
void
HeightFeatsMiner<XDecimalType, FDecimalType>::computeHeightKurtosis(
    arma::Row<XDecimalType> const &xi,
    arma::Col<FDecimalType> const &z,
    bool &minComputed,
    bool &maxComputed,
    bool &meanComputed,
    bool &stdevComputed,
    FDecimalType &min,
    FDecimalType &max,
    FDecimalType &mean,
    FDecimalType &stdev,
    arma::uword &nextFeatIdx,
    arma::subview_row<FDecimalType> &out
){
    // Compute mean if it has not been computed before
    if(!meanComputed){
        mean = arma::mean(z);
        meanComputed = true;
    }
    // Compute stdev if it has not been computed before
    if(!stdevComputed){
        stdev = arma::stddev(z);
        stdevComputed = true;
    }
    // Compute the skewness
    if(stdev < 1e-6){
        out[nextFeatIdx] = 0.0;
    }
    else {
        arma::Col<FDecimalType> const y = (z - mean) / stdev;
        out[nextFeatIdx] = arma::mean(y % y % y) - 3.0;
    }
    ++nextFeatIdx;
}
