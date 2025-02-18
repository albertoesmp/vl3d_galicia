// ***   INLCUDES   *** //
// ******************** //
#include <adt/grid/SparseGrid.hpp>

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType, typename IndexType>
SparseGrid<XDecimalType, IndexType>::SparseGrid(
    XDecimalType const size,
    IndexType const paddingCells,
    int const nthreads
) :
    size(size),
    paddingCells(paddingCells),
    nthreads(nthreads),
    logTime(true)
{
    // Handle as many threads as available
    if(this->nthreads==-1) this->nthreads = std::thread::hardware_concurrency();
}

template <typename XDecimalType, typename IndexType>
SparseGrid<XDecimalType, IndexType>::SparseGrid(
    XDecimalType const size,
    arma::Row<XDecimalType> const &A,
    arma::Col<IndexType> const &n
) :
    size(size),
    paddingCells(0),
    nthreads(1),
    A(A),
    n(n),
    logTime(false)
{}

template <typename XDecimalType, typename IndexType>
SparseGrid<XDecimalType, IndexType>::SparseGrid(
    XDecimalType const size,
    arma::Row<XDecimalType> const &A,
    arma::Col<IndexType> const &n,
    arma::Col<IndexType> const &hk,
    arma::Col<IndexType> const &hv
) :
    size(size),
    paddingCells(0),
    nthreads(1),
    A(A),
    n(n),
    logTime(false)
{
    // Build map h
    arma::uword const m = hk.n_rows;
    for(arma::uword i = 0 ; i < m ; ++i){
        h.emplace(hk[i], hv[i]);
    }
}


// ***  MAIN METHODS  *** //
// ********************** //
template <typename XDecimalType, typename IndexType>
void
SparseGrid<XDecimalType, IndexType>::fit(arma::Mat<XDecimalType> const &X){
    // Start measuring time
    TimeWatcher tw;
    if(logTime) tw.start();
    // Prepare computations
    arma::uword const m = X.n_rows; // Number of input points
    // Find min and max vertices
    A = arma::min(X, 0);
    arma::Row<XDecimalType> B = arma::max(X, 0);
    // Apply padding, if requested
    if(paddingCells > 0){
        A -= paddingCells*size;
        B += paddingCells*size;
    }
    // Prepare grid
    arma::Row<XDecimalType> D = B-A; // Length for each axis
    n = arma::conv_to<arma::Col<IndexType>>::from(
        arma::ceil(D/size) // ceil of D/l (element-wise division)
    );
    h = std::move(map<IndexType, IndexType>());
    // Build grid
    for(arma::uword i = 0, k = 0 ; i < m ; ++i){
        IndexType const gi = indexFromCoordinates(X.row(i));
        if(h.find(gi) == h.end()){ // If gi was not activated before
            h.emplace(gi, k++); // Register it as active
        }
    }
    // Report execution time
    if(logTime){
        tw.stop();
        std::stringstream ss;
        arma::uword nCells = n[0];
        ss  << "SparseGrid of " << n[0];
        for(arma::uword i = 1 ; i < n.n_rows ; ++i){
            ss << " x " << n[i];
            nCells *= n[i];
        }
        ss  << " (" << nCells << ") cells with " << h.size() << " active cells "
            << "fit to " << m << " points in " << tw.getElapsedDecimalSeconds()
            << " seconds."
        ;
        LOGGER->logInfo(ss.str());
    }
}

template <typename XDecimalType, typename IndexType>
template <typename FType>
arma::Mat<FType>
SparseGrid<XDecimalType, IndexType>::encodeMatrix(
    arma::Mat<XDecimalType> const &X,
    arma::Mat<FType> const &F,
    string const &reduceStrategy
){
    // Prepare computations
    arma::uword const nf = F.n_cols; // Number of features
    arma::uword const R = h.size(); // Number of active cells
    FType (*reduce) (vector<FType> const &f);
    vector<vector<IndexType>> I = prepareEncoding(
        X,
        F,
        reduceStrategy,
        &reduce
    );
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most
    // Cell-wise reduce
    arma::Mat<FType> Y(R, nf);
    #pragma omp parallel for default(none) shared(R, I, nf, F, Y, reduce) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED)
    for(arma::uword i = 0 ; i < R ; ++i){ // Iterate i-th active cell
        vector<IndexType> const &Ii = I[i];
        vector<FType> vals(Ii.size());
        for(arma::uword j = 0 ; j < nf ; ++j){ // Iterate j-th feature
            for(arma::uword k = 0 ; k < Ii.size() ; ++k){ // k-th point in cell
                vals[k] = F.at(Ii[k], j);
            }
            Y.at(i, j) = reduce(vals);
        }
    }
    // Return cell-wise features for the R active cells only
    return Y;
}

template <typename XDecimalType, typename IndexType>
template <typename FType>
arma::Col<FType>
SparseGrid<XDecimalType, IndexType>::encodeVector(
    arma::Mat<XDecimalType> const &X,
    arma::Col<FType> const &F,
    string const &reduceStrategy
){
    // Prepare computations
    arma::uword const R = h.size(); // Number of active cells
    FType (*reduce) (vector<FType> const &f);
    vector<vector<IndexType>> I = prepareEncoding(
        X,
        F,
        reduceStrategy,
        &reduce
    );
    omp_set_num_threads(nthreads); // Use nthreads parallel threads at most
    // Cell-wise reduce
    arma::Col<FType> Y(R);
    #pragma omp parallel for default(none) shared(R, I, F, Y, reduce) \
        schedule(VL3DPP_OMP_SCHEDULE_CHUNKED)
    for(arma::uword i = 0 ; i < R ; ++i){ // Iterate i-th active cell
        vector<IndexType> const &Ii = I[i];
        vector<FType> vals(Ii.size());
        for(arma::uword k = 0 ; k < Ii.size() ; ++k){ // k-th point in cell
            vals[k] = F[Ii[k]];
        }
        Y[i] = reduce(vals);
    }
    // Return cell-wise features for the R active cells only
    return Y;
}

// ***  REDUCE METHODS  *** //
// ************************ //
template <typename XDecimalType, typename IndexType>
template <typename FType>
FType
SparseGrid<XDecimalType, IndexType>::reduceMean(vector<FType> const &f){
    if(f.size() == 1) return f[0];
    FType mean = 0;
    for(FType const &fi : f) mean += fi;
    return mean / ((FType) f.size());
}

template <typename XDecimalType, typename IndexType>
template <typename FType>
FType
SparseGrid<XDecimalType, IndexType>::reduceMax(vector<FType> const &f){
    if(f.size() == 1) return f[0];
    FType max = std::numeric_limits<FType>::lowest();
    for(FType const &fi : f) max = std::max(max, fi);
    return max;
}

template <typename XDecimalType, typename IndexType>
template <typename FType>
FType
SparseGrid<XDecimalType, IndexType>::reduceMin(vector<FType> const &f){
    if(f.size() == 1) return f[0];
    FType min = std::numeric_limits<FType>::max();
    for(FType const &fi : f) min = std::min(min, fi);
    return min;
}

template <typename XDecimalType, typename IndexType>
template <typename FType>
FType
SparseGrid<XDecimalType, IndexType>::reduceMode(vector<FType> const &f){
    if(f.size() == 1) return f[0];
    // Prepare
    map<FType, IndexType> recount;
    // Count samples per value
    for(FType const &fi : f){
        typename map<FType, IndexType>::iterator it =
            recount.find(fi);
        if(it == recount.end()){ // First sample
            recount[fi] = 1; // Initialize recount to one
        }
        else{ // After first sample
            it->second += 1; // Increase recount in one
        }
    }
    // Choose value with max number of samples
    FType modeVal = 0;
    IndexType maxCount = 0;
    typename map<FType, IndexType>::const_iterator itEnd =
        recount.cend();
    for(
        typename map<FType, IndexType>::const_iterator it =
            recount.cbegin();
        it != itEnd;
        ++it
    ){
        FType const val = it->first;
        IndexType const count = it->second;
        if(count > maxCount){
            modeVal = val;
            maxCount = count;
        }
    }
    // Return mode value
    return modeVal;
}


// ***  UTIL METHODS  *** //
// ********************** //
template <typename XDecimalType, typename IndexType>
IndexType
SparseGrid<XDecimalType, IndexType>::indexFromCoordinates(
    arma::Row<XDecimalType> const &xi
){
    // Prepare
    IndexType factor = 1; // Index scale factor (initialized for last dimension)
    IndexType const nx = xi.n_cols; // Dimensionality
    IndexType gi = 0; // Cell index for given point xi (initialized to zero)
    // Iterate from last dimension to first one (not inclusive)
    for(size_t d = nx-1 ; d > 0 ; --d){
        gi += std::min( // Compute contribution of d-th dimension to the index
            (IndexType) (n[d]-1),
            (IndexType) ((xi[d]-A[d])/size)
        ) * factor;
        factor *= n[d]; // Update factor to scale next contribution
    }
    // Compute contribution of first dimension to the index
    gi += std::min(
        (IndexType) (n[0]-1),
        (IndexType) ((xi[0]-A[0])/size)
    ) * factor;
    // Return cell index
    return gi;
}

template <typename XDecimalType, typename IndexType>
template <typename FType>
vector<vector<IndexType>>
SparseGrid<XDecimalType, IndexType>::prepareEncoding(
    arma::Mat<XDecimalType> const &X,
    arma::Mat<FType> const &F,
    string const &reduceStrategy,
    FType (**reduce) (vector<FType> const &f)
){
    // Prepare computations
    arma::uword const m = X.n_rows; // Number of input points
    arma::uword const R = h.size(); // Number of active cells
    arma::uword const baseCapacity = std::max( // Initial mem for each vector
        32,
        (int) std::ceil(X.n_rows / R)
    );
    vector<vector<IndexType>> I( // Indices for all values
        R // i.e., before reduce
    ); // I[i] gives the indices of the points inside the i-th active cell
    if(reduceStrategy == "mean"){
        *reduce = &SparseGrid<XDecimalType, IndexType>::reduceMean;
    }
    else if(reduceStrategy == "max"){
        *reduce = &SparseGrid<XDecimalType, IndexType>::reduceMax;
    }
    else if(reduceStrategy == "min"){
        *reduce = &SparseGrid<XDecimalType, IndexType>::reduceMin;
    }
    else if(reduceStrategy == "mode"){
        *reduce = &SparseGrid<XDecimalType, IndexType>::reduceMode;
    }
    else{
        std::stringstream ss;
        ss  << "SparseGrid::prepareEncoding failed to determine reduce "
            << "function due to an unexpected reduce strategy (\") "
            << reduceStrategy << "\")";
        throw vl3dpp::util::VL3DPPException(ss.str());
    }
    for(vector<IndexType> &Ii : I) Ii.reserve(baseCapacity);
    // Find all values for each cell
    for(arma::uword i = 0 ; i < m ; ++i){
        IndexType const gi = indexFromCoordinates(X.row(i));
        I[h[gi]].push_back(i);
    }
    // Return indices of points in active cells
    return I;
}


template <typename XDecimalType, typename IndexType>
void SparseGrid<XDecimalType, IndexType>::report(std::ostream & out){
    out << "size: " << size << "\n"
        << "paddingCells: " << paddingCells << "\n"
        << "nthreads: " << nthreads << "\n"
        << "logTime: " << logTime << "\n"
        << "h:\n"
        << "\tsize: " << h.size() << "\n"
    ;
    if(h.size() > 6){
        size_t const mh = h.size();
        std::vector<IndexType> hk(6), hv(6);
        size_t i = 0, j = 0;
        for(auto it = h.cbegin() ; it != h.cend() ; ++it){
            if( i < 3 || i >= (mh-3)){
                hk[j] = it->first;
                hv[j] = it->second;
                ++j;
            }
            ++i;
        }
        out << "\t(hk1, hk2, hk3, ..., hk-3, hk-2, hk-1) = ("
            << hk[0] << ", " << hk[1] << ", " << hk[2] << ", ..., "
            << hk[3] << ", " << hk[4] << ", " << hk[5] << ")\n"
            << "\t(hv1, hv2, hv3, ..., hv-3, hv-2, hv-1) = ("
            << hv[0] << ", " << hv[1] << ", " << hv[2] << ", ..., "
            << hv[3] << ", " << hv[4] << ", " << hv[5] << ")"
            << std::endl
        ;
    }
    out << "A:\n"
        << "\tdimensionality: " << A.n_cols << "\n"
        << "\tvalues: (" << std::fixed
    ;
    for(arma::uword i = 0 ; i < A.n_cols ; ++i){
        out << A[i] << (i==A.n_cols-1 ? ")\n" : ", ");
    }
    out << "n:\n"
        <<  "\tdimensionality: " << n.n_rows << "\n"
        << "\tvalues: " << n.as_row()
    ;
}
