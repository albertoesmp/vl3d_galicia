#include <alg/FurthestPointSubsampler.hpp>


// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType>
FurthestPointSubsampler<XDecimalType>::FurthestPointSubsampler(
    arma::uword const targetPoints,
    int const nthreads
) :
    targetPoints(targetPoints),
    nthreads(nthreads)
{}



// ***   MAIN METHODS   *** //
// ************************ //
template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::sample(
    arma::Mat<XDecimalType> const &X
) const{
    // Check target points can be reached
    arma::uword const m = X.n_rows; // Number of input points
    if(m < targetPoints){
        std::stringstream ss;
        ss  << "FurthestPointSubsampler cannot subsample " << m << " points "
            << "into " << targetPoints << " points because " << m << " < "
            << targetPoints;
        throw VL3DPPException(ss.str());
    }

    // Prepare exhaustive FPS computation
    arma::uword const nx = X.n_cols; // Dimensionality of structure space
    arma::Col<arma::uword> selected(targetPoints); // Points from X to FPS
    arma::Col<XDecimalType> minSqDist( // Vector of min/closest distances^2
        m, arma::fill::value(std::numeric_limits<XDecimalType>::max())
    );

    // Iteratively find all points for FPS
    arma::uword fp = 0; // Index of furthest point for each iteration
    selected[0] = 0;  // Select first point
    minSqDist[0] = 0; // First point will be considered, so distance zero
    for(arma::uword R=1 ; R < targetPoints ; ++R){ // R is num. points in FPS
        // Find furthest point wrt to current selection
        XDecimalType maxSqDist = 0;
        arma::Row<XDecimalType> const &fpx = X.row(fp); // Coords. of furth. pt.
        for(arma::uword i = 1 ; i < m ; ++i){ // Iterate over points
            XDecimalType sqDist = 0;
            for(arma::uword j = 0 ; j < nx ; ++j){
                XDecimalType const diff = X.at(i, j) - fpx[j];
                sqDist += diff*diff;
            }
            minSqDist[i] = std::min(
                sqDist,
                minSqDist[i]
            );
            if(minSqDist[i] > maxSqDist){
                fp = i;
                maxSqDist = minSqDist[i];
            }
        }
        // Register furthest point
        selected[R] = fp;
    }

    // Return FPS
    return X.rows(selected);
}

template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::sample2D(
    arma::Mat<XDecimalType> const &X
) const{
    // Check target points can be reached
    arma::uword const m = X.n_rows; // Number of input points
    if(m < targetPoints){
        std::stringstream ss;
        ss  << "FurthestPointSubsampler cannot subsample " << m << " 2D "
            << "points into " << targetPoints << " points because " << m
            << " < " << targetPoints;
        throw VL3DPPException(ss.str());
    }

    // Prepare exhaustive FPS computation
    arma::Col<arma::uword> selected(targetPoints); // Points from X to FPS
    arma::Col<XDecimalType> minSqDist( // Vector of min/closest distances^2
        m, arma::fill::value(std::numeric_limits<XDecimalType>::max())
    );
    // Iteratively find all points for FPS
    arma::uword fp = 0; // Index of furthest point for each iteration
    selected[0] = 0;  // Select first point
    minSqDist[0] = 0; // First point will be considered, so distance zero
    for(arma::uword R=1 ; R < targetPoints ; ++R){ // R is num. points in FPS
        // Find furthest point wrt to current selection
        XDecimalType maxSqDist = 0;
        arma::Row<XDecimalType> const &fpx = X.row(fp); // Coords. of furth. pt.
        for(arma::uword i = 1 ; i < m ; ++i){ // Iterate over points
            XDecimalType const dx = X.at(i, 0)-fpx[0];
            XDecimalType const dy = X.at(i, 1)-fpx[1];
            minSqDist[i] = std::min(dx*dx+dy*dy, minSqDist[i]);
            if(minSqDist[i] > maxSqDist){
                fp = i;
                maxSqDist = minSqDist[i];
            }
        }
        // Register furthest point
        selected[R] = fp;
    }

    // Return FPS
    return X.rows(selected);
}

template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::sample3D(
    arma::Mat<XDecimalType> const &X
) const{
    // Check target points can be reached
    arma::uword const m = X.n_rows; // Number of input points
    if(m < targetPoints){
        std::stringstream ss;
        ss  << "FurthestPointSubsampler cannot subsample " << m << " 3D "
            << "points into " << targetPoints << " points because " << m
            << " < " << targetPoints;
        throw VL3DPPException(ss.str());
    }

    // Prepare exhaustive FPS computation
    arma::Col<arma::uword> selected(targetPoints); // Points from X to FPS
    arma::Col<XDecimalType> minSqDist( // Vector of min/closest distances^2
        m, arma::fill::value(std::numeric_limits<XDecimalType>::max())
    );

    // Iteratively find all points for FPS
    arma::uword fp = 0; // Index of furthest point for each iteration
    selected[0] = 0;  // Select first point
    minSqDist[0] = 0; // First point will be considered, so distance zero
    for(arma::uword R=1 ; R < targetPoints ; ++R){ // R is num. points in FPS
        // Find furthest point wrt to current selection
        XDecimalType maxSqDist = 0;
        arma::Row<XDecimalType> const &fpx = X.row(fp); // Coords. of furth. pt.
        for(arma::uword i = 1 ; i < m ; ++i){ // Iterate over points
            XDecimalType const dx = X.at(i, 0)-fpx[0];
            XDecimalType const dy = X.at(i, 1)-fpx[1];
            XDecimalType const dz = X.at(i, 2)-fpx[2];
            minSqDist[i] = std::min(dx*dx+dy*dy+dz*dz, minSqDist[i]);
            if(minSqDist[i] > maxSqDist){
                fp = i;
                maxSqDist = minSqDist[i];
            }
        }
        // Register furthest point
        selected[R] = fp;
    }

    // Return FPS
    return X.rows(selected);
}


// ***  PARALLEL STOCHASTIC FPS METHODS  *** //
// ***************************************** //
template <typename XDecimalType>
arma::Mat<XDecimalType>
FurthestPointSubsampler<XDecimalType>::parallelSample(
    arma::Mat<XDecimalType> const &X
) const{
    // Check target points can be reached
    arma::uword const m = X.n_rows; // Number of input points
    if(m < targetPoints){
        std::stringstream ss;
        ss  << "FurthestPointSubsampler cannot subsample " << m << " points "
            << "into " << targetPoints << " points because " << m << " < "
            << targetPoints;
        throw VL3DPPException(ss.str());
    }

    // Prepare exhaustive FPS computation
    arma::uword const nx = X.n_cols; // Dimensionality of structure space
    arma::Col<arma::uword> selected(targetPoints); // Points from X to FPS
    arma::Col<XDecimalType> minSqDist( // Vector of min/closest distances^2
        m, arma::fill::value(std::numeric_limits<XDecimalType>::max())
    );

    // Iteratively find all points for FPS
    selected[0] = 0;  // Select first point
    minSqDist[0] = 0; // First point will be considered, so distance zero
    VL3DPP_OMP_Opt<XDecimalType, arma::uword> fp = {0, 0};
    int const chunkSize = util::MultithreadingUtils::computeFullStaticChunkSize(
        m-1, nthreads
    );
    #pragma omp parallel default(none) \
        shared(targetPoints, selected, fp, m, X, minSqDist, chunkSize)
    { // OMP parallel begin ---
    for(arma::uword R=1 ; R < targetPoints ; ++R){ // R is num. points in FPS
        // Find furthest point wrt to current selection
        fp.x = 0;
        arma::Row<XDecimalType> const &fpx = X.row(fp.i); // Coords. of furth.
        #pragma omp for reduction(argmax:fp) schedule(static, chunkSize)
        for(arma::uword i = 1 ; i < m ; ++i){ // Iterate over points
            XDecimalType sqDist = 0;
            for(arma::uword j = 0 ; j < nx ; ++j){
                XDecimalType const diff = X.at(i, j) - fpx[j];
                sqDist += diff*diff;
            }
            minSqDist[i] = std::min(
                sqDist,
                minSqDist[i]
            );
            if(minSqDist[i] > fp.x){
                fp.x = minSqDist[i];
                fp.i = i;
            }
        }
        // Register furthest point
        selected[R] = fp.i;
    }
    } // --- OMP parallel end

    // Return FPS
    return X.rows(selected);
}

template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::parallelSample2D(
    arma::Mat<XDecimalType> const &X
) const{
    // Check target points can be reached
    arma::uword const m = X.n_rows; // Number of input points
    if(m < targetPoints){
        std::stringstream ss;
        ss  << "FurthestPointSubsampler cannot subsample " << m << " 2D "
            << "points into " << targetPoints << " points because " << m
            << " < " << targetPoints;
        throw VL3DPPException(ss.str());
    }

    // Prepare exhaustive FPS computation
    arma::Col<arma::uword> selected(targetPoints); // Points from X to FPS
    arma::Col<XDecimalType> minSqDist( // Vector of min/closest distances^2
        m, arma::fill::value(std::numeric_limits<XDecimalType>::max())
    );

    // Iteratively find all points for FPS
    selected[0] = 0;  // Select first point
    minSqDist[0] = 0; // First point will be considered, so distance zero
    VL3DPP_OMP_Opt<XDecimalType, arma::uword> fp = {0, 0};
    int const chunkSize = util::MultithreadingUtils::computeFullStaticChunkSize(
        m-1, nthreads
    );
    #pragma omp parallel default(none) \
        shared(targetPoints, selected, fp, m, X, minSqDist, chunkSize)
    { // OMP parallel begin ---
    for(arma::uword R=1 ; R < targetPoints ; ++R){ // R is num. points in FPS
        #pragma omp single
        { // OMP single begin ---
        // Find furthest point wrt to current selection
        fp.x = 0;
        } // --- OMP single end
        arma::Row<XDecimalType> const &fpx = X.row(fp.i); // Coords. of furth. pt.
        #pragma omp for reduction(argmax:fp) schedule(static, chunkSize)
        for(arma::uword i = 1 ; i < m ; ++i){
            // Iterate over points
            XDecimalType const dx = X.at(i, 0)-fpx[0];
            XDecimalType const dy = X.at(i, 1)-fpx[1];
            minSqDist[i] = std::min(dx*dx+dy*dy, minSqDist[i]);
            if(minSqDist[i] > fp.x){
                fp.x = minSqDist[i];
                fp.i = i;
            }
        }
        #pragma omp single
        { // OMP single begin ---
        // Register furthest point
        selected[R] = fp.i;
        } // --- OMP single end
    }
    } // --- OMP parallel end
    // Return FPS
    return X.rows(selected);
}

template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::parallelSample3D(
    arma::Mat<XDecimalType> const &X
) const{
    // Check target points can be reached
    arma::uword const m = X.n_rows; // Number of input points
    if(m < targetPoints){
        std::stringstream ss;
        ss  << "FurthestPointSubsampler cannot subsample " << m << " 3D "
            << "points into " << targetPoints << " points because " << m
            << " < " << targetPoints;
        throw VL3DPPException(ss.str());
    }

    // Prepare exhaustive FPS computation
    arma::Col<arma::uword> selected(targetPoints); // Points from X to FPS
    arma::Col<XDecimalType> minSqDist( // Vector of min/closest distances^2
        m, arma::fill::value(std::numeric_limits<XDecimalType>::max())
    );

    // Iteratively find all points for FPS
    selected[0] = 0;  // Select first point
    minSqDist[0] = 0; // First point will be considered, so distance zero
    VL3DPP_OMP_Opt<XDecimalType, arma::uword> fp = {0, 0};
    int const chunkSize = util::MultithreadingUtils::computeFullStaticChunkSize(
        m-1, nthreads
    );
    #pragma omp parallel default(none) \
        shared(targetPoints, selected, fp, m, X, minSqDist, chunkSize)
    { // OMP parallel begin ---
    for(arma::uword R=1 ; R < targetPoints ; ++R){ // R is num. points in FPS
        #pragma omp single
        { // OMP single begin ---
        // Find furthest point wrt to current selection
        fp.x = 0;
        } // --- OMP single end
        arma::Row<XDecimalType> const &fpx = X.row(fp.i); // Coords. of furth. pt.
        #pragma omp for reduction(argmax:fp) schedule(static, chunkSize)
        for(arma::uword i = 1 ; i < m ; ++i){
            // Iterate over points
            XDecimalType const dx = X.at(i, 0)-fpx[0];
            XDecimalType const dy = X.at(i, 1)-fpx[1];
            XDecimalType const dz = X.at(i, 2)-fpx[2];
            minSqDist[i] = std::min(dx*dx+dy*dy+dz*dz, minSqDist[i]);
            if(minSqDist[i] > fp.x){
                fp.x = minSqDist[i];
                fp.i = i;
            }
        }
        #pragma omp single
        { // OMP single begin ---
        // Register furthest point
        selected[R] = fp.i;
        } // --- OMP single end
    }
    } // --- OMP parallel end

    // Return FPS
    return X.rows(selected);
}

// ***  HYBRID STOCHASTIC FPS METHODS  *** //
// *************************************** //
template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::hybridSample(
    arma::Mat<XDecimalType> const &X
) const {
    // Uniform downsampling
    arma::Mat<XDecimalType> Xsup = overDownSample(X);
    // Do exact FPS on downsampled points, if necessary
    if(Xsup.n_rows != targetPoints){
        return sample(Xsup);
    }
    // If not necessary, straightforward return
    return Xsup;
}

template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::hybridSample2D(
    arma::Mat<XDecimalType> const &X
) const {
    // Uniform downsampling
    arma::Mat<XDecimalType> Xsup = overDownSample(X);
    // Do exact FPS on downsampled points, if necessary
    if(Xsup.n_rows != targetPoints){
        return sample2D(Xsup);
    }
    // If not necessary, straightforward return
    return Xsup;
}

template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::hybridSample3D(
    arma::Mat<XDecimalType> const &X
) const{
    // Uniform downsampling
    arma::Mat<XDecimalType> Xsup = overDownSample(X);
    // Do exact FPS on downsampled points, if necessary
    if(Xsup.n_rows != targetPoints){
        return sample3D(Xsup);
    }
    // If not necessary, straightforward return
    return Xsup;
}

template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::overDownSample(
    arma::Mat<XDecimalType> const &X
) const{
    // Uniform downsampling (not necessarily fixed to targetPints)
    arma::uword const iStep = X.n_rows / targetPoints;
    arma::uword const numDownsampledPoints = X.n_rows / iStep;
    arma::Mat<XDecimalType> Xsup(numDownsampledPoints, X.n_cols);
    for(arma::uword i = 0 ; i < numDownsampledPoints ; ++i){
        Xsup.row(i) = X.row(i*iStep);
    }
    return Xsup;
}

// ***  FULL STOCHASTIC FPS METHODS  *** //
// ************************************* //
template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::downSample(
    arma::Mat<XDecimalType> const &X
) const{
    // Check if necessary
    if(X.n_rows == targetPoints) return X;

    // Uniform downsampling
    arma::uword const iStep = X.n_rows / targetPoints;
    arma::Mat<XDecimalType> Xsup(targetPoints, X.n_cols);
    for(arma::uword i = 0 ; i < targetPoints ; ++i){
        Xsup.row(i) = X.row(i*iStep);
    }
    return Xsup;
}

template <typename XDecimalType>
arma::Mat<XDecimalType> FurthestPointSubsampler<XDecimalType>::randomSample(
    arma::Mat<XDecimalType> const &X
) const{
    arma::Col<arma::uword> const indices = arma::shuffle(
        arma::regspace<arma::Col<arma::uword>>(0, X.n_rows-1)
    );
    return X.rows(indices.subvec(0, targetPoints-1));
}
