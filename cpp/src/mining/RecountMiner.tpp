#include <mining/RecountMiner.hpp>
#include <math/MathConstants.hpp>
#include <util/VL3DPPException.hpp>
#include <util/TimeWatcher.hpp>

#include <algorithm>

using vl3dpp::mining::RecountMiner;
using vl3dpp::util::VL3DPPException;
using vl3dpp::util::TimeWatcher;


// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType, typename FDecimalType>
RecountMiner<XDecimalType, FDecimalType>::RecountMiner(
    string const &nbhType,
    int const nbhK,
    arma::Col<XDecimalType> const nbhRadius,
    XDecimalType const nbhLowerBound,
    XDecimalType const nbhUpperBound,
    std::vector<bool> const &ignoreNan,
    std::vector<bool> const &absFreq,
    std::vector<bool> const &relFreq,
    std::vector<bool> const &surfDens,
    std::vector<bool> const &volDens,
    std::vector<int> const &vertSeg,
    std::vector<int> const &rings,
    std::vector<int> const &radBound,
    std::vector<int> const &sect2D,
    std::vector<int> const &sect3D,
    std::vector<std::vector<arma::uword>> const &condFeatIndices,
    std::vector<std::vector<std::string>> const &condTypes,
    std::vector<std::vector<std::vector<FDecimalType>>> const &condTargets,
    int const nthreads
) :
    nbhType(nbhType),
    nbhK(nbhK),
    nbhRadius(nbhRadius),
    nbhLowerBound(nbhLowerBound),
    nbhUpperBound(nbhUpperBound),
    ignoreNan(ignoreNan),
    absFreq(absFreq),
    relFreq(relFreq),
    surfDens(surfDens),
    volDens(volDens),
    vertSeg(vertSeg),
    rings(rings),
    radBound(radBound),
    sect2D(sect2D),
    sect3D(sect3D),
    condFeatIndices(condFeatIndices),
    condTypes(condTypes),
    condTargets(condTargets),
    nthreads(nthreads)
{
    // Handle number of threads -1
    if(nthreads == -1) this->nthreads = std::thread::hardware_concurrency();
    // Assign NaN handling method
    for(bool const ignore : this->ignoreNan){
        if(ignore){
            nanHandlingFunctions.push_back([] (
                arma::Mat<XDecimalType> &XN,
                arma::Mat<FDecimalType> &FN
            ) -> void {
                std::vector<arma::uword> nanRows;
                nanRows.reserve(FN.n_rows);
                for(arma::uword i = 0 ; i < FN.n_rows ; ++i){
                    if(FN.row(i).has_nan()) nanRows.push_back(i);
                }
                arma::Col<arma::uword> const _nanRows =
                    arma::conv_to<arma::Col<arma::uword>>::from(nanRows);
                XN.shed_rows(_nanRows);
                FN.shed_rows(_nanRows);
            });
        }
        else{
            nanHandlingFunctions.push_back([] (
                arma::Mat<XDecimalType> &XN,
                arma::Mat<FDecimalType> &FN
            ) -> void {
                return;
            });
        }
    }
    // Assign neighborhood query function
    if(nbhType == "knn"){
        query = [] (
            arma::subview_row<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return kdt.findKnn(xi, k);
        };
    }
    else if(nbhType == "knn2d"){
        query = [] (
            arma::subview_row<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return kdt.findKnn2D(xi, k);
        };
    }
    else if(nbhType == "sphere"){
        query = [] (
            arma::subview_row<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return kdt.findSphere(xi, radius[0]);
        };
    }
    else if(nbhType == "cylinder"){
        query = [] (
            arma::subview_row<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return kdt.findCylinder(xi, radius[0]);
        };
    }
    else if(nbhType == "rectangular3d"){
        query = [] (
            arma::subview_row<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return kdt.findBox(xi, radius);
        };
    }
    else if(nbhType == "rectangular2d"){
        query = [] (
            arma::subview_row<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return kdt.findRectangle(xi, radius);
        };
    }
    else if(nbhType == "boundedcylinder"){
        query = [] (
            arma::subview_row<XDecimalType> const &xi,
            arma::subview_col<XDecimalType> const &z,
            KDTree<arma::uword, XDecimalType> &kdt,
            arma::uword const k,
            arma::Col<XDecimalType> const &radius,
            XDecimalType const lowerBound,
            XDecimalType const upperBound
        ) -> arma::Col<arma::uword> {
            return kdt.findBoundedCylinder(
                xi, z, radius[0], lowerBound, upperBound
            );
        };
    }
    // Assign condition handling functions
    for(size_t i = 0 ; i < condTypes.size() ; ++i){
        std::vector<std::string> const &types = condTypes[i];
        std::vector<std::function<void(
            arma::Mat<XDecimalType> &XN,
            arma::Mat<FDecimalType> &FN,
            arma::uword const fIdx,
            std::vector<FDecimalType> const &target
        )>> condFuns;
        for(size_t j = 0 ; j < types.size() ; ++j){
            std::string type = types[j];
            std::transform( // Type to lower case
                type.begin(), type.end(), type.begin(),
                [](unsigned char c) {return std::tolower(c);}
            );
            if(type == "not_equals"){
                condFuns.push_back(notEqualsCondFun);
            }
            else if(type == "equals"){
                condFuns.push_back(equalsCondFun);
            }
            else if(type == "less_than"){
                condFuns.push_back(lessThanCondFun);
            }
            else if(type == "less_than_or_equal_to"){
                condFuns.push_back(lessThanOrEqualToCondFun);
            }
            else if(type == "greater_than"){
                condFuns.push_back(greaterThanCondFun);
            }
            else if(type == "greater_than_or_equal_to"){
                condFuns.push_back(greaterThanOrEqualToCondFun);
            }
            else if(type == "in"){
                condFuns.push_back(inCondFun);
            }
            else if(type == "not_in"){
                condFuns.push_back(notInCondFun);
            }
            else{
                std::stringstream ss;
                ss  << "RecountMiner++ received an unexpected condition type: "
                    << "\"" << types[j] << "\"";
                throw VL3DPPException(ss.str());
            }
        }
        conditionFunctions.push_back(condFuns);
    }
    // Assign recount functions
    for(size_t i = 0 ; i < absFreq.size() ; ++i){
        std::vector<std::function<FDecimalType(
            arma::subview_row<XDecimalType> const &xi,
            arma::Mat<XDecimalType> const &XN,
            arma::Mat<FDecimalType> const &FN,
            XDecimalType r,
            int const K
        )>> recounts;
        std::vector<XDecimalType> r;
        std::vector<int> K;
        if(absFreq[i]){
            recounts.push_back(recountAbsoluteFrequency);
            r.push_back(0);
            K.push_back(0);
        }
        if(relFreq[i]){
            recounts.push_back(recountRelativeFrequency);
            r.push_back(0);
            K.push_back(-1);
        }
        if(surfDens[i]){
            recounts.push_back(recountSurfaceDensity);
            if(nbhType == "knn" || nbhType == "knn2d") r.push_back(0);
            else r.push_back(arma::max(nbhRadius));
            K.push_back(0);
        }
        if(volDens[i]){
            recounts.push_back(recountVolumeDensity);
            if(nbhType == "knn" || nbhType == "knn2d") r.push_back(0);
            else r.push_back(arma::max(nbhRadius));
            if(
                nbhType == "knn2d" || nbhType == "cylinder" ||
                nbhType=="rectangular2d" || nbhType=="boundedcylinder"
            ) K.push_back(2);
        }
        if(vertSeg[i] > 0){
            recounts.push_back(recountVerticalSegments);
            r.push_back(0);
            K.push_back(vertSeg[i]);
        }
        if(rings[i] > 0){
            recounts.push_back(recountRings);
            if(nbhType == "knn" || nbhType == "knn2d") r.push_back(0);
            else r.push_back(arma::max(nbhRadius));
            K.push_back(rings[i]);
        }
        if(radBound[i] > 0){
            recounts.push_back(recountRadialBoundaries);
            if(nbhType == "knn" || nbhType == "knn2d") r.push_back(0);
            else r.push_back(arma::max(nbhRadius));
            K.push_back(radBound[i]);
        }
        if(sect2D[i] > 0){
            recounts.push_back(recountSectors2D);
            r.push_back(0);
            K.push_back(sect2D[i]);
        }
        if(sect3D[i] > 0){
            recounts.push_back(recountSectors3D);
            r.push_back(0);
            K.push_back(sect3D[i]);
        }
        recountFunctions.push_back(recounts);
        recountRadius.push_back(r);
        recountK.push_back(K);
    }
    // Number of output features
    numFeatures = 0;
    for(size_t i = 0 ; i < recountFunctions.size() ; ++i){
        numFeatures += recountFunctions[i].size();
    }
}


// ***  DATA MINING METHODS  *** //
// ***************************** //
template <typename XDecimalType, typename FDecimalType>
arma::Mat<FDecimalType> RecountMiner<XDecimalType, FDecimalType>::mine(
    arma::Mat<XDecimalType> const &X,
    arma::Mat<FDecimalType> const &F
){
    // Start global time measurement
    TimeWatcher twGlobal; twGlobal.start();

    // Build KDTree
    TimeWatcher twKDT; twKDT.start();
    bool kdt3D = true;
    if(
        nbhType == "knn2d" || nbhType == "cylinder" ||
        nbhType == "rectangular2d" || nbhType == "boundedcylinder"
    ) kdt3D = false;
    KDTree<arma::uword, XDecimalType> kdt(X, !kdt3D, kdt3D);
    twKDT.stop();
    std::stringstream ss;
    ss  << "RecountMinerPP built the KDTree for " << X.n_rows << " points "
        << "in " << twKDT.getElapsedDecimalSeconds() << " seconds.";
    LOGGER->logInfo(ss.str());

    // Prepare recount-based features computation
    TimeWatcher twComp; twComp.start();
    arma::uword const m = X.n_rows; // Number of points
    arma::Mat<FDecimalType> Fhat(m, numFeatures);
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most

    // Compute point-wise recount-based features
    #pragma omp parallel for default(none) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED) \
        shared(X, F, kdt, m, Fhat)
    for(arma::uword i = 0 ; i < m ; ++i){ // Recounts for each point
        computeRecounts(X.row(i), i, X, F, kdt, Fhat.row(i));
    }

    // Report recount-based features computation
    twComp.stop();
    ss.str("");
    ss  << "RecountMinerPP computed the " << m << " x " << numFeatures
        << " matrix of recount-based features in "
        << twComp.getElapsedDecimalSeconds() << " seconds.";
    LOGGER->logInfo(ss.str());

    // Report total time
    twGlobal.stop();
    ss.str("");
    ss  << "RecountMinerPP finished in " << twGlobal.getElapsedDecimalSeconds()
        << " seconds.";
    LOGGER->logInfo(ss.str());

    // Return
    return Fhat;
}

template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::computeRecounts(
    arma::subview_row<XDecimalType> const &xi,
    arma::uword const i,
    arma::Mat<XDecimalType> const &X,
    arma::Mat<FDecimalType> const &F,
    KDTree<arma::uword, XDecimalType> &kdt,
    arma::subview_row<FDecimalType> fout
) const{
    // Find the neighborhood
    arma::Col<arma::uword> const N = query(
        xi,
        X.col(2),
        kdt,
        nbhK,
        nbhRadius,
        nbhLowerBound,
        nbhUpperBound
    );
    // Compute each filter
    arma::uword j = 0;
    for(size_t k = 0 ; k < recountFunctions.size() ; ++k){
        // Extract neighborhoods
        arma::Mat<XDecimalType> XN = X.rows(N);
        arma::Mat<FDecimalType> FN = F.rows(N);
        // Apply NaN policy
        nanHandlingFunctions[k](XN, FN);
        int const K = XN.n_rows; // Number of points before condition-based filtering
        for(size_t l = 0 ; l < conditionFunctions[k].size() ; ++l) {
            // Apply condition-based filters
            conditionFunctions[k][l](
                XN, FN, condFeatIndices[k][l], condTargets[k][l]
            );
        }
        for(size_t l = 0 ; l < recountFunctions[k].size() ; ++l, ++j){
            // Compute the recount-based features
            fout[j] = recountFunctions[k][l](
                xi,
                XN,
                FN,
                recountRadius[k][l],
                (recountK[k][l]==-1) ? K : recountK[k][l]
            );
        }
    }
}


// ***   CONDITION METHODS   *** //
// ***************************** //
template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::notEqualsCondFun(
    arma::Mat<XDecimalType> &XN,
    arma::Mat<FDecimalType> &FN,
    arma::uword const fIdx,
    std::vector<FDecimalType> const &target
){
    arma::subview_col<FDecimalType> const f = FN.col(fIdx);
    arma::Col<arma::uword> const preserveIndices = arma::find(f != target[0]);
    XN = XN.rows(preserveIndices);
    FN = FN.rows(preserveIndices);
}

template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::equalsCondFun(
    arma::Mat<XDecimalType> &XN,
    arma::Mat<FDecimalType> &FN,
    arma::uword const fIdx,
    std::vector<FDecimalType> const &target
){
    arma::subview_col<FDecimalType> const f = FN.col(fIdx);
    arma::Col<arma::uword> const preserveIndices = arma::find(f == target[0]);
    XN = XN.rows(preserveIndices);
    FN = FN.rows(preserveIndices);
}

template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::lessThanCondFun(
    arma::Mat<XDecimalType> &XN,
    arma::Mat<FDecimalType> &FN,
    arma::uword const fIdx,
    std::vector<FDecimalType> const &target
){
    arma::subview_col<FDecimalType> const f = FN.col(fIdx);
    arma::Col<arma::uword> const preserveIndices = arma::find(f < target[0]);
    XN = XN.rows(preserveIndices);
    FN = FN.rows(preserveIndices);
}

template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::lessThanOrEqualToCondFun(
    arma::Mat<XDecimalType> &XN,
    arma::Mat<FDecimalType> &FN,
    arma::uword const fIdx,
    std::vector<FDecimalType> const &target
){
    arma::subview_col<FDecimalType> const f = FN.col(fIdx);
    arma::Col<arma::uword> const preserveIndices = arma::find(f <= target[0]);
    XN = XN.rows(preserveIndices);
    FN = FN.rows(preserveIndices);
}

template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::greaterThanCondFun(
    arma::Mat<XDecimalType> &XN,
    arma::Mat<FDecimalType> &FN,
    arma::uword const fIdx,
    std::vector<FDecimalType> const &target
){
    arma::subview_col<FDecimalType> const f = FN.col(fIdx);
    arma::Col<arma::uword> const preserveIndices = arma::find(f > target[0]);
    XN = XN.rows(preserveIndices);
    FN = FN.rows(preserveIndices);
}

template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::greaterThanOrEqualToCondFun(
    arma::Mat<XDecimalType> &XN,
    arma::Mat<FDecimalType> &FN,
    arma::uword const fIdx,
    std::vector<FDecimalType> const &target
){
    arma::subview_col<FDecimalType> const f = FN.col(fIdx);
    arma::Col<arma::uword> const preserveIndices = arma::find(f >= target[0]);
    XN = XN.rows(preserveIndices);
    FN = FN.rows(preserveIndices);
}

template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::inCondFun(
    arma::Mat<XDecimalType> &XN,
    arma::Mat<FDecimalType> &FN,
    arma::uword const fIdx,
    std::vector<FDecimalType> const &target
){
    arma::subview_col<FDecimalType> const f = FN.col(fIdx);
    std::vector<bool> preserveMask(FN.n_rows, false);
    for(FDecimalType const iTarget : target){
        arma::Col<arma::uword> const inIndices = arma::find(FN == iTarget);
        for(arma::uword k : inIndices) preserveMask[k] = true;
    }
    size_t preserveCount = 0;
    for(bool const preserve : preserveMask) if(preserve) ++preserveCount;
    arma::Col<arma::uword> preserveIndices(preserveCount);
    for(arma::uword i = 0, j = 0  ; i < FN.n_rows ; ++i){
        if(preserveMask[i]){
            preserveIndices[j] = i;
            ++j;
        }
    }
    XN = XN.rows(preserveIndices);
    FN = FN.rows(preserveIndices);
}

template <typename XDecimalType, typename FDecimalType>
void RecountMiner<XDecimalType, FDecimalType>::notInCondFun(
    arma::Mat<XDecimalType> &XN,
    arma::Mat<FDecimalType> &FN,
    arma::uword const fIdx,
    std::vector<FDecimalType> const &target
){

    arma::subview_col<FDecimalType> const f = FN.col(fIdx);
    std::vector<bool> preserveMask(FN.n_rows, true);
    for(FDecimalType const iTarget : target){
        arma::Col<arma::uword> const inIndices = arma::find(FN == iTarget);
        for(arma::uword k : inIndices) preserveMask[k] = false;
    }
    size_t preserveCount = 0;
    for(bool const preserve : preserveMask) if(preserve) ++preserveCount;
    arma::Col<arma::uword> preserveIndices(preserveCount);
    for(arma::uword i = 0, j = 0  ; i < FN.n_rows ; ++i){
        if(preserveMask[i]){
            preserveIndices[j] = i;
            ++j;
        }
    }
    XN = XN.rows(preserveIndices);
    FN = FN.rows(preserveIndices);
}


// ***   RECOUNT METHODS   *** //
// *************************** //
template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountAbsoluteFrequency(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    return (FDecimalType) FN.n_rows;
}

template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountRelativeFrequency(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    return (K != 0) ? (FN.n_rows / ((FDecimalType)K)) : 0;
}

template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountSurfaceDensity(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    // Handle no points case
    if(FN.n_rows == 0) return 0;
    // Compute radius from neighborhood, if necessary
    if(r == 0){
        r = std::sqrt(arma::max(arma::sum(
            arma::square(XN.cols(0, 1)-xi.cols(0, 1)), 1
        )));
    }
    // Compute the surface density
    return FN.n_rows/(M_PI*r*r);
}

template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountVolumeDensity(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    // Handle no points case
    if(FN.n_rows == 0) return 0;
    // Handle cylindrical case (specified through K=2)
    if(K==2){
        // Compute radius from neighborhood, if necessary
        if(r == 0){
            r = std::sqrt(arma::max(arma::sum(
                arma::square(XN.cols(0, 1)-xi.cols(0, 1)), 1
            )));
        }
        arma::subview_col<XDecimalType> const z = XN.col(2);
        XDecimalType const zmin = arma::min(z);
        XDecimalType const zmax = arma::max(z);
        XDecimalType const zdelta = zmax-zmin;
        XDecimalType const boundedCylinderVolume = M_PI*r*r*zdelta;
        if(zdelta == 0) return std::numeric_limits<FDecimalType>::max();
        return ((FDecimalType) FN.n_rows) / boundedCylinderVolume;
    }
    // Compute radius from neighborhood, if necessary
    if(r == 0) r = std::sqrt(arma::max(arma::sum(arma::square(XN-xi), 1)));
    // Compute the volume density assuming a spherical neighborhood
    return ((FDecimalType)FN.n_rows) / (4.0*M_PI*r*r*r/3.0);

}

template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountVerticalSegments(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    // Handle no points case
    if(FN.n_rows == 0) return 0;
    // Find cut points separating vertical segments
    arma::subview_col<XDecimalType> const z = XN.col(2);
    XDecimalType const zmin = arma::min(z);
    XDecimalType const zmax = arma::max(z);
    arma::Col<XDecimalType> const cuts =
        arma::linspace<arma::Col<XDecimalType>>(zmin, zmax, K);
    // Analyze the first segment
    int count = arma::any(z <= cuts[0]) ? 1 : 0;
    for(int i = 1 ; i < K ; ++i){
        if(arma::any((z > cuts[i-1]) % (z <= cuts[i]))) ++count;
    }
    // Return count of vertical segments with at least one point
    return (FDecimalType) count;
}

template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountRings(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    // Handle no points case
    if(FN.n_rows == 0) return 0;
    // Compute 2D square distances
    arma::Col<XDecimalType> sqD(XN.n_rows);
    for(size_t i = 0 ; i < XN.n_rows ; ++i){
        sqD[i] = arma::sum(arma::square(XN.row(i).cols(0, 1) - xi.cols(0, 1)));
    }
    // Compute radius from neighborhood, if necessary
    if(r == 0) r = std::sqrt(arma::max(sqD));
    // Count occupied rings
    arma::Col<XDecimalType> const R = arma::linspace<arma::Col<XDecimalType>>(
        0, r, K+1
    );
    int count = arma::any(sqD <= R[1]) ? 1 : 0;
    for(int i = 1 ; i < K ; ++i){
        if(arma::any((sqD > R[i]*R[i]) % (sqD <= R[i+1]*R[i+1]))) ++count;
    }
    // Return count of occupied rings
    return count;
}

template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountRadialBoundaries(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    // Handle no points case
    if(FN.n_rows == 0) return 0;
    // Compute 2D square distances
    arma::Col<XDecimalType> sqD(XN.n_rows);
    for(size_t i = 0 ; i < XN.n_rows ; ++i){
        sqD[i] = arma::sum(arma::square(XN.row(i) - xi));
    }
    // Compute radius from neighborhood, if necessary
    if(r == 0) r = std::sqrt(arma::max(sqD));
    // Count occupied rings
    arma::Col<XDecimalType> const R = arma::linspace<arma::Col<XDecimalType>>(
        0, r, K+1
    );
    int count = arma::any(sqD <= R[1]) ? 1 : 0;
    for(int i = 1 ; i < K ; ++i){
        if(arma::any((sqD > R[i]*R[i]) % (sqD <= R[i+1]*R[i+1]))){
            ++count;
        }
    }
    // Return count of occupied rings
    return count;
}

template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountSectors2D(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    // Handle no points case
    if(FN.n_rows == 0) return 0;
    // Compute angles on the plane (2D)
    arma::Col<XDecimalType> const theta = arma::atan2(
        XN.col(1)-xi[1],
        XN.col(0)-xi[0]
    );
    // Count populated 2D sectors
    arma::Col<XDecimalType> const secTheta = arma::linspace<
        arma::Col<XDecimalType>
    >(-M_PI, M_PI, K+1);
    int count = arma::any(theta <= secTheta[1]);
    for(int i =  1;  i < K ; ++i ){
        if(arma::any((theta > secTheta[i]) % (theta <= secTheta[i+1])))
            ++count;
    }
    // Return count of occupied 2D sectors
    return count;
}

template <typename XDecimalType, typename FDecimalType>
FDecimalType RecountMiner<XDecimalType, FDecimalType>::recountSectors3D(
    arma::subview_row<XDecimalType> const &xi,
    arma::Mat<XDecimalType> const &XN,
    arma::Mat<FDecimalType> const &FN,
    XDecimalType r,
    int const K
){
    // Handle no points case
    if(FN.n_rows == 0) return 0;
    // Compute elevation (K1) and horizontal cuts (K2)
    arma::uword const K1 = (arma::uword) std::ceil(std::sqrt(K));
    arma::uword const K2 = (arma::uword) std::ceil(K/K1);
    // Compute polar (elevation cuts, phi) and azimuth angles (plane, theta)
    arma::Col<XDecimalType> phi(FN.n_rows);
    arma::Col<XDecimalType> theta(FN.n_rows);
    arma::Row<XDecimalType> const e3({0, 0, 1});
    for(arma::uword i = 0; i < FN.n_rows ; ++i){
        arma::Row<XDecimalType> const u = XN.row(i)-xi;
        if(arma::all(u==0)) continue; // Skip xi
        XDecimalType const uNorm = arma::norm(u);
        phi[i] = (uNorm == 0) ? 0 : std::acos(arma::dot(e3, u)/uNorm);
        theta[i] = std::atan2(u[1], u[0]);
    }
    // Count populated 3D sectors
    arma::Col<XDecimalType> const phiCuts = arma::linspace<
        arma::Col<XDecimalType>
    >(0, M_PI, K1+1);
    arma::Col<XDecimalType> const thetaCuts = arma::linspace<
        arma::Col<XDecimalType>
    >(-M_PI, M_PI, K2+1);
    int count = 0; // Count populated 3D sectors
    for(arma::uword i = 0 ; i < K1 ; ++i){ // Polar sectors
        arma::Col<arma::uword> const inPolar = (i > 0) ?
            ((phi > phiCuts[i]) % (phi <= phiCuts[i+1])).eval() :
            (phi <= phiCuts[i+1]).eval()
        ;
        for(arma::uword j = 0 ; j < K2 ; ++j){ // Azimuth sectors
            arma::Col<arma::uword> const inAzimuth = (i > 0) ?
                ((theta > thetaCuts[j]) % (theta <= thetaCuts[j+1])).eval() :
                (theta <= thetaCuts[j+1]).eval()
            ;
            if(arma::any(inPolar % inAzimuth)) ++count;
        }
    }
    // Return count of occupied 3D sectors
    return count;
}
