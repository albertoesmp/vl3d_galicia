#include <alg/Oversampler.hpp>


// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType, typename IndexType>
Oversampler<XDecimalType, IndexType>::Oversampler(
     IndexType const minPoints,
     IndexType const targetPoints,
     std::string const &strategy,
     IndexType const K,
     XDecimalType const radius
) :
    minPoints(minPoints),
    targetPoints(targetPoints),
    strategy(strategy),
    K(K),
    radius(radius){
    // Determine oversampling method from strategy
    if(this->strategy == "nearest"){
        this->oversample = &Oversampler::nearestOversample;
    }
    else if(this->strategy == "knn"){
        this->oversample = &Oversampler::knnOversample;
    }
    else if(this->strategy == "gaussian_knn"){
        this->oversample = &Oversampler::gaussianKnnOversample;
    }
    else if(this->strategy == "spherical"){
        this->oversample = &Oversampler::sphericalOversample;
    }
    else if(this->strategy == "spherical_radiation"){
        this->oversample = &Oversampler::sphericalRadiationOversample;
    }
    else{
        std::stringstream ss;
        ss  << "Oversampler cannot be instantiated due to an unexpected "
            << "strategy: \"" << this->strategy << "\"";
        throw vl3dpp::util::VL3DPPException(ss.str());
    }
}

// ***   CALLABLE   *** //
// ******************** //
template <typename XDecimalType, typename IndexType>
arma::Mat<XDecimalType> Oversampler<XDecimalType, IndexType>::operator()(
    arma::Mat<XDecimalType> const &X
) const{
    // Check that oversampling is needed
    if(X.n_rows >= targetPoints) return X;
    // Check that the minimum number of points is met
    if(X.n_rows < minPoints){
        std::stringstream ss;
        ss  << "Oversampler cannot oversample " << X.n_rows << " to "
            << targetPoints << " points because the minimum number of points "
            << minPoints << "is not reached.";
        throw VL3DPPException(ss.str());
    }
    // Do the oversampling
    IndexType const kappa = targetPoints - X.n_rows;
    return (this->*oversample)(kappa, X);
}


// ***  OVERSAMPLING METHODS  *** //
// ****************************** //
template <typename XDecimalType, typename IndexType>
arma::Mat<XDecimalType> Oversampler<XDecimalType, IndexType>::nearestOversample(
    IndexType const kappa,
    arma::Mat<XDecimalType> const &X
) const{
    // Prepare nearest oversampling
    arma::Mat<XDecimalType> Y(targetPoints, X.n_cols);
    Y.rows(0, X.n_rows-1) = X;
    IndexType currentPoints = X.n_rows;
    IndexType Knext =  // How many points oversample in the first iteration
        (targetPoints > currentPoints) ?
            std::min(currentPoints, targetPoints-currentPoints) :
            0
    ;
    // Compute nearest oversampling
    do{
        // Find pairs of closest neighbors
        KDTree<IndexType, XDecimalType> kdt(
            Y.rows(0, currentPoints-1), false, true
        );
        arma::Mat<IndexType> I(currentPoints, 2); // Neighborhood indices
        arma::Mat<XDecimalType> D(currentPoints, 2); // Neighborhood distances
        kdt.findKnn(
            (arma::Mat<XDecimalType>) Y.rows(0, currentPoints-1),
            (IndexType) 2,
            I,
            D
        );
        // Sort indices by ascending order when not all pairs will be included
        if(Knext < currentPoints){
            arma::Col<arma::uword> const S = arma::sort_index(
                (arma::Col<XDecimalType>) D.col(0)
            );
            I = I.rows(S);
        }
        // Oversample one point for each considered pair of closest neighbors
        for(arma::uword i = 0 ; i < Knext ; ++i){
            arma::Row<XDecimalType> const &p = Y.row(I.at(i, 0));
            arma::Row<XDecimalType> const &q = Y.row(I.at(i, 1));
            Y.row(currentPoints+i) = (p+q)/2.0;
        }
        // How many points there are right now
        currentPoints += Knext;
        // How many points must be oversampled in the next iteration
        Knext =
            (targetPoints > currentPoints) ?
                std::min(currentPoints, targetPoints-currentPoints) :
                0
        ;
    } while(Knext > 0);
    // Return oversampled points
    return Y;
}

template <typename XDecimalType, typename IndexType>
arma::Mat<XDecimalType> Oversampler<XDecimalType, IndexType>::knnOversample(
    IndexType const kappa,
    arma::Mat<XDecimalType> const &X
) const{
    // Prepare nearest oversampling
    arma::Mat<XDecimalType> Y(targetPoints, X.n_cols);
    Y.rows(0, X.n_rows-1) = X;
    IndexType currentPoints = X.n_rows;
    IndexType Knext =  // How many points oversample in the first iteration
        (targetPoints > currentPoints) ?
            std::min(currentPoints, targetPoints-currentPoints) :
            0
    ;
    // Compute nearest oversampling
    do{
        // Find pairs of closest neighbors
        KDTree<IndexType, XDecimalType> kdt(
            Y.rows(0, currentPoints-1), false, true
        );
        arma::Mat<IndexType> I(currentPoints, K); // Neighborhood indices
        arma::Mat<XDecimalType> D(currentPoints, K); // Neighborhood distances
        kdt.findKnn(
            (arma::Mat<XDecimalType>) Y.rows(0, currentPoints-1),
            K,
            I,
            D
        );
        // Sort indices by ascending order when not all pairs will be included
        if(Knext < currentPoints){
            arma::Col<arma::uword> const S = arma::sort_index(
                (arma::Col<XDecimalType>) D.col(D.n_cols-1)
            );
            I = I.rows(S);
        }
        // Oversample one point for each considered k-nearest neighborhood
        for(arma::uword i = 0 ; i < Knext ; ++i){
            arma::Row<XDecimalType> p = Y.row(I.at(i, 0));
            for(arma::uword k = 1 ; k < K ; ++k) p += Y.row(I.at(i, k));
            Y.row(currentPoints+i) = p/((XDecimalType) K);
        }
        // How many points there are right now
        currentPoints += Knext;
        // How many points must be oversampled in the next iteration
        Knext =
            (targetPoints > currentPoints) ?
                std::min(currentPoints, targetPoints-currentPoints) :
                0
        ;
    } while(Knext > 0);
    // Return oversampled points
    return Y;
}

template <typename XDecimalType, typename IndexType>
arma::Mat<XDecimalType>
Oversampler<XDecimalType, IndexType>::gaussianKnnOversample(
    IndexType const kappa,
    arma::Mat<XDecimalType> const &X
) const{
    // Prepare nearest oversampling
    arma::Mat<XDecimalType> Y(targetPoints, X.n_cols);
    Y.rows(0, X.n_rows-1) = X;
    IndexType currentPoints = X.n_rows;
    IndexType Knext =  // How many points oversample in the first iteration
        (targetPoints > currentPoints) ?
            std::min(currentPoints, targetPoints-currentPoints) :
            0
    ;
    // Compute nearest oversampling
    do{
        // Find pairs of closest neighbors
        KDTree<IndexType, XDecimalType> kdt(
            Y.rows(0, currentPoints-1), false, true
        );
        arma::Mat<IndexType> I(currentPoints, K); // Neighborhood indices
        arma::Mat<XDecimalType> D(currentPoints, K); // Neighborhood distances
        kdt.findKnn(
            (arma::Mat<XDecimalType>) Y.rows(0, currentPoints-1),
            K,
            I,
            D
        );
        // Sort indices by ascending order when not all pairs will be included
        if(Knext < currentPoints){
            arma::Col<arma::uword> const S = arma::sort_index(
                (arma::Col<XDecimalType>) D.col(D.n_cols-1)
            );
            I = I.rows(S);
        }
        // Oversample one point for each considered k-nearest neighborhood
        for(arma::uword i = 0 ; i < Knext ; ++i){
            // Prepare i-th neighborhood oversampling
            arma::Row<XDecimalType> p(X.n_cols, arma::fill::zeros);
            // Compute centroid for i-th neighborhood
            arma::Col<arma::uword> const Ii = arma::conv_to<
                arma::Col<arma::uword>
            >::from(I.row(i));
            arma::Row<XDecimalType> const mu = arma::mean(Y.rows(Ii));
            XDecimalType dSqMax = 0.0; // Max squared distance
            // Compute squared distances wrt centroid (reusing D)
            for(arma::uword k = 0 ; k < K ; ++k){ // For each k-th neighbor
                D.at(i, k) = arma::sum(arma::square(Y.row(I.at(i, k)) - mu));
                if(D.at(i, k) > dSqMax) dSqMax = D.at(i, k);
            }
            // Aggregate neighborhood coordinates weighting with Gaussian RBF
            XDecimalType norm = 0; // Norm for the Gaussian RBF superposition
            for(arma::uword k = 0 ; k < K ; ++k){
                // Compute square distance wrt centroid for k-th neighbor
                // Compute weight as Gaussian RBF
                XDecimalType const w = std::exp(-D.at(i, k)/dSqMax);
                p += w*Y.row(I.at(i, k));
                norm += w;
            }
            Y.row(currentPoints+i) = p/norm;
        }
        // How many points there are right now
        currentPoints += Knext;
        // How many points must be oversampled in the next iteration
        Knext =
            (targetPoints > currentPoints) ?
                std::min(currentPoints, targetPoints-currentPoints) :
                0
        ;
    } while(Knext > 0);
    // Return oversampled points
    return Y;
}

template <typename XDecimalType, typename IndexType>
arma::Mat<XDecimalType>
Oversampler<XDecimalType, IndexType>::sphericalOversample(
    IndexType const kappa,
    arma::Mat<XDecimalType> const &X
) const{
    // Prepare nearest oversampling
    arma::Mat<XDecimalType> Y(targetPoints, X.n_cols);
    Y.rows(0, X.n_rows-1) = X;
    IndexType currentPoints = X.n_rows;
    IndexType Knext =  // How many points oversample in the first iteration
        (targetPoints > currentPoints) ?
            std::min(currentPoints, targetPoints-currentPoints) :
            0
    ;
    // Compute nearest oversampling
    do{
        // Find pairs of closest neighbors
        KDTree<IndexType, XDecimalType> kdt(
            Y.rows(0, currentPoints-1), false, true
        );
        std::vector<arma::Col<IndexType>> I = // Neighborhood indices
            kdt.findSphere(
                (arma::Mat<XDecimalType>) Y.rows(0, currentPoints-1),
                radius
            );
        // Oversample one point for each considered spherical neighborhood
        for(arma::uword i = 0 ; i < Knext ; ++i){
            arma::Col<arma::uword> const Ii =
                arma::conv_to<arma::Col<arma::uword>>::from(I[i]);
            Y.row(currentPoints+i) = arma::mean(Y.rows(Ii), 0);
        }
        // How many points there are right now
        currentPoints += Knext;
        // How many points must be oversampled in the next iteration
        Knext =
            (targetPoints > currentPoints) ?
                std::min(currentPoints, targetPoints-currentPoints) :
                0
        ;
    } while(Knext > 0);
    // Return oversampled points
    return Y;
}

template <typename XDecimalType, typename IndexType>
arma::Mat<XDecimalType>
Oversampler<XDecimalType, IndexType>::sphericalRadiationOversample(
    IndexType const kappa,
    arma::Mat<XDecimalType> const &X
) const{
    // Prepare nearest oversampling
    arma::Mat<XDecimalType> Y(targetPoints, X.n_cols);
    Y.rows(0, X.n_rows-1) = X;
    IndexType currentPoints = X.n_rows;
    IndexType Knext =  // How many points oversample in the first iteration
        (targetPoints > currentPoints) ?
            std::min(currentPoints, targetPoints-currentPoints) :
            0
    ;
    XDecimalType const sqRadius = radius*radius;
    // Compute nearest oversampling
    do{
        // Find pairs of closest neighbors
        KDTree<IndexType, XDecimalType> kdt(
            Y.rows(0, currentPoints-1), false, true
        );
        std::vector<arma::Col<IndexType>> I; // Neighborhood indices
        std::vector<arma::Col<XDecimalType>> D; // Neighborhood distances
        kdt.findSphere(
            (arma::Mat<XDecimalType>) Y.rows(0, currentPoints-1),
            radius,
            I,
            D
        );
        // Oversample one point for each considered spherical neighborhood
        for(arma::uword i = 0 ; i < Knext ; ++i){
            // Prepare i-th neighborhood oversampling
            arma::Row<XDecimalType> p(X.n_cols, arma::fill::zeros);
            // Compute centroid for i-th neighborhood
            arma::Col<arma::uword> const Ii = arma::conv_to<
                arma::Col<arma::uword>
            >::from(I[i]);
            arma::Col<XDecimalType> &Di = D[i];
            arma::Row<XDecimalType> const mu = arma::mean(Y.rows(Ii));
            XDecimalType dSqMax = 0.0; // Max squared distance
            // Compute squared distances wrt centroid (reusing D)
            for(arma::uword k = 0 ; k < Ii.n_rows ; ++k){ // For each neighbor
                Di[k] = arma::sum(arma::square(Y.row(Ii[k]) - mu));
                if(Di[k] > dSqMax) dSqMax = Di[k];
            }
            // Aggregate neighborhood coordinates weighting with Gaussian RBF
            XDecimalType norm = 0; // Norm for the Gaussian RBF superposition
            for(arma::uword k = 0 ; k < Ii.n_rows ; ++k){
                // Compute square distance wrt centroid for k-th neighbor
                // Compute weight as Gaussian RBF
                XDecimalType const w = std::exp(-Di[k]/sqRadius);
                p += w*Y.row(Ii[k]);
                norm += w;
            }
            Y.row(currentPoints+i) = p/norm;
        }
        // How many points there are right now
        currentPoints += Knext;
        // How many points must be oversampled in the next iteration
        Knext =
            (targetPoints > currentPoints) ?
                std::min(currentPoints, targetPoints-currentPoints) :
                0
        ;
    } while(Knext > 0);
    // Return oversampled points
    return Y;
}
