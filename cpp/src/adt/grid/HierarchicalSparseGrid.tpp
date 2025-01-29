// ***   INCLUDES   *** //
// ******************** //
#include <adt/grid/HierarchicalSparseGrid.hpp>

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
template <typename XDecimalType, typename IndexType>
HierarchicalSparseGrid<XDecimalType, IndexType>::HierarchicalSparseGrid(
    XDecimalType const size,
    arma::Col<IndexType> const &w,
    arma::Col<IndexType> const &wD,
    arma::Col<IndexType> const &wU,
    arma::Col<IndexType> const &sD,
    arma::Col<IndexType> const &sU,
    int const nthreads
) :
    w(w),
    wD(wD),
    wU(wU),
    sD(sD),
    sU(sU),
    logTime(false),
    sg(size, 0, nthreads)
{
    // Check max depth is correct (same for window sizes and strides)
    if(wD.n_elem != sD.n_elem){ // Downsampling sizes
        std::stringstream ss;
        ss  << "HierarchicalSparseGrid instantiation failed due to an "
            << "ambiguous max depth. " << wD.n_elem << " downsampling "
            << "window sizes were given but " << sD.n_elem << " "
            << "downsampling strides were given."
        ;
        throw vl3dpp::util::VL3DPPException(ss.str());
    }
    if(wU.n_elem != sU.n_elem){ // Upsampling sizes
        std::stringstream ss;
        ss  << "HierarchicalSparseGrid instantiation failed due to an "
            << "ambiguous max depth. " << wU.n_elem << " upsampling "
            << "window sizes were given but " << sU.n_elem << " "
            << "upsampling strides were given."
        ;
        throw vl3dpp::util::VL3DPPException(ss.str());
    }
    if(wD.n_elem != wU.n_elem){ // Downsampling VS upsampling sizes
        std::stringstream ss;
        ss  << "HierarchicalSparseGrid instantiation failed due to an "
            << "ambiguous max depth. " << wD.n_elem << " downsampling "
            << "window sizes were given but " << wU.n_elem << " "
            << "upsampling window sizes were given."
        ;
        throw vl3dpp::util::VL3DPPException(ss.str());
    }
    if(wD.n_elem != w.n_elem-1){ // Downsampling VS submanifold sizes
        std::stringstream ss;
        ss  << "HierarchicalSparseGrid instantiation failed due to an "
            << "ambiguous max depth. " << wD.n_elem << " downsampling "
            << "window sizes were given but " << w.n_elem << " "
            << "submanifold window sizes were given. "
            << "The number of submanifold window sizes is the max depth. "
            << "Thus, it must be exactly one plus the number of downsampling "
            << "window sizes."
        ;
        throw vl3dpp::util::VL3DPPException(ss.str());
    }
}


// ***  MAIN METHODS  *** //
// ********************** //
template <typename XDecimalType, typename IndexType>
void HierarchicalSparseGrid<XDecimalType, IndexType>::fit(
    arma::Mat<XDecimalType> const &X
){
    // Measure start time
    TimeWatcher tw;
    if(logTime) tw.start();
    // Prepare hierarchy
    IndexType const maxDepth = getMaxDepth();
    IndexType const nx = X.n_cols;
    h.resize(maxDepth-1);
    hD.resize(maxDepth-1);
    hU.resize(maxDepth-1);
    n.set_size(nx, maxDepth-1);
    // Build initial sparse grid
    buildFirstSparseGrid(X, nx);
    // Build hierarchy of sparse grid after the first one
    for(IndexType t = 1 ; t < maxDepth ; ++t){ // Iterate depths after first
        buildHierarchicalSparseGrid(t, maxDepth, nx);
    }
    // Report execution time
    if(logTime){
        tw.stop();
        std::stringstream ss;
        ss  << "HierarchicalSparseGrid of depth " << maxDepth << " fit to "
            << X.n_rows << " points in " << tw.getElapsedDecimalSeconds() << " "
            << "seconds"
        ;
        LOGGER->logInfo(ss.str());
    }
}


// ***  HIERARCHY BUILDING METHODS  *** //
// ************************************ //
template <typename XDecimalType, typename IndexType>
void
HierarchicalSparseGrid<XDecimalType, IndexType>::buildFirstSparseGrid(
    arma::Mat<XDecimalType> const &X,
    IndexType const nx
){
    // Compute expected number of axis partitions without padding
    arma::Row<XDecimalType> A = arma::min(X, 0); // Min vertex
    arma::Row<XDecimalType> B = arma::max(X, 0); // Max vertex
    arma::Row<XDecimalType> D = B-A; // Axis-wise range
    arma::Col<IndexType> const nSg = arma::conv_to<arma::Col<IndexType>>::from(
        arma::ceil(D/sg.getCellSize()) // ceil of D/l (divide for each axis)
    );
    // Update padding of first sparse grid to fit in hierarchy
    sg.setPaddingCells(computeHalfPadding(
        nx,
        nSg,
        getSubmanifoldWindow(0),
        getDownsamplingWindow(0),
        getDownsamplingStride(0)
    ));
    // Fit the first sparse grid
    sg.setLogTime(getLogTime()); // Propagate logTime policy before fit
    sg.fit(X);
    // Reorder first hash map (h, h0, h[0]) to be sequential on the values
    map<IndexType, IndexType> &h = sg.getMap();
    typename map<IndexType, IndexType>::iterator it;
    IndexType i = 0;
    for(it = h.begin() ; it != h.end() ; ++it){
        it->second = i++;
    }
}

template <typename XDecimalType, typename IndexType>
void
HierarchicalSparseGrid<XDecimalType, IndexType>::buildHierarchicalSparseGrid(
    IndexType const t, // depth (t), must be greater than zero
    IndexType const tmax, // max depth (t^*), must be greater than one
    IndexType const nx // Structure space dimensionality
){
    // Determine number of cells of sparse grid at depth t, considering padding
    IndexType const wt = getSubmanifoldWindow(t);
    IndexType const wDt = getDownsamplingWindow(t-1);
    IndexType const sDt = getDownsamplingStride(t-1);
    IndexType const wUt = getUpsamplingWindow(t-1);
    IndexType const sUt = getUpsamplingStride(t-1);
    arma::Col<IndexType> const nSource = getNumAxisPartitions(t-1);
    arma::Col<IndexType> mt(nx); // Number of partitions without padding
    for(IndexType d = 0 ; d < nx ; ++d){
        mt[d] = (nSource[d]-wDt+sDt) / sDt;
    }
    IndexType const pt = computeHalfPadding(nx, mt, wt, wDt, sDt);
    for(IndexType d = 0 ; d < nx ; ++d){
        n.at(d, t-1) = mt[d] + 2*pt; // d-th axis partitions
    }
    // Determine active cells of sparse grid at depth t
    map<IndexType, IndexType> const &hSource = getMap(t-1);
    map<IndexType, IndexType> &ht = getMap(t);
    std::vector<IndexType> &hDt = getDownsamplingMap(t-1);  hDt.clear();
    std::vector<IndexType> &hUt = getUpsamplingMap(t-1);    hUt.clear();
    IndexType j = 0; // Index to iterate over current grid cells
    IndexType kj = 0; // Index to iterate over current grid active cells
    arma::Col<IndexType> k(nx, arma::fill::zeros); // Cell-wise indices
    IndexType wDtonx = 1; // (w^D)^{n_x}, i.e, nx-times wD
    IndexType wUtonx = 1; // (w^U)^{n_x}, i.e., nx-times wU
    arma::Col<IndexType> eod(nx); // End-of-dimension (EOD) thresholds
    IndexType i0 = 0; // Starting index for i at current depth
    for(IndexType d = 0 ; d < nx ; ++d){
        eod[d] = nSource[d]-wDt+1;
        wDtonx *= wDt;
        wUtonx *= wUt;
        IndexType const i0Left= ((int)((nSource[d]-wDt+sDt) % sDt)/2);
        IndexType i0Right = 1;
        for(IndexType l=d+1 ; l < nx ; ++l){
            i0Right *= nSource[l];
        }
        i0 += i0Left*i0Right;
    }
    IndexType i = i0; // Index to iterate over source cells
    // Precompute products
    IndexType pnProd = 1;
    arma::Col<IndexType> dTermPreCache(nx-1);
    arma::Col<IndexType> dTermPostCache(nx-1);
    arma::Col<IndexType> vtds(nx);
    arma::Col<IndexType> nlSourceProds(nx);
    arma::Col<IndexType> vtlProds(nx);
    arma::Col<IndexType> prodDownsampleds(nx);
    for(IndexType d = 0 ; d < nx ; ++d){
        vtds[d] = (IndexType) ((n.at(d, t-1) - wUt + sUt)/sUt);
        nlSourceProds[d] = 1;
        vtlProds[d] = 1;
        prodDownsampleds[d] = 1;
        for(IndexType l = d+1 ; l < nx ; ++l){
            nlSourceProds[d] *= nSource[l];
            vtlProds[d] *= (IndexType) ((n.at(l, t-1) - wUt + sUt)/sUt);
            prodDownsampleds[d] *= n.at(l, t-1);
        }
    }
    for(IndexType d = 1 ; d < nx ; ++d){
        pnProd *= n.at(d, t-1);
        IndexType dTermPre = 1, dTermPost = 1;
        for(IndexType l = d ; l < nx ; ++l){
            dTermPre *= mt[l];
        }
        for(IndexType l = d+1 ; l < nx ; ++l){
            dTermPost *= n.at(l, t-1);
        }
        dTermPreCache[d-1] = dTermPre;
        dTermPostCache[d-1] = dTermPost;
    }
    pnProd *= pt;
    do{ // Compute submanifold and downsampling indexing
        // Check at least one active cell inside window at current position
        bool jActive = false;
        for(IndexType wk = 0 ; wk < wDtonx ; ++wk){
            IndexType wi = i; // Cell on source grid inside window
            for(IndexType d = 0 ; d < nx ; ++d){
                IndexType wDivisor=1, sourceFactor=1;
                for(IndexType l = d+1 ; l < nx ; ++l){
                    wDivisor *= wDt;
                    sourceFactor *= nSource[l];
                }
                wi += (IndexType)(wk/wDivisor) % wDt * sourceFactor;
            }
            if(hSource.find(wi) != hSource.end()){ // If source active cell
                jActive = true; // The current grid cell is also active
                break; // No more iterations
            }
        }
        // If so, the cell in the downsampled sparse grid is active
        if(jActive){
            // Transform j from domain without padding to domain with padding
            IndexType padSum = 0;
            for(IndexType d = 1 ; d < nx ; ++d){
                IndexType dTerm = dTermPreCache[d-1];
                dTerm = pt*(1 + 2*((IndexType)j/dTerm));
                dTerm *= dTermPostCache[d-1];
                padSum += dTerm;
            }
            IndexType const jAux = j + pnProd + padSum; // j considering padding
            ht.emplace(jAux, kj); // Submanifold indexing
            hDt.push_back(i); // Downsampling indexing
            ++kj; // Index for next active cell in current grid
        }
        ++j; // Index for next cell in current grid
        // Move window to next position
        for(IndexType d = nx ; d > 0 ; --d){ // Find next cell-wise indices
            IndexType const kd = k[d-1]+sDt;
            if(kd >= eod[d-1]){
                k[d-1] = 0;
                if(d==1) break; // First axis repeating zero implies last i
            }
            else{ // Forward on current dimension
                k[d-1] = kd;
                break;
            }
        }
        i = i0;
        for(IndexType d = 0 ; d < nx ; ++d){ // Find next i
            IndexType id = k[d];
            for(IndexType l = d+1 ; l < nx ; ++l){
                id *= nSource[l];
            }
            i += id;
        }
    } while(i!=i0);
    // Compute  usapmling indexing
    typename map<IndexType, IndexType>::const_iterator it;
    for(it=hSource.cbegin(); it!=hSource.cend() ; ++it){
        i = it->first;
        j = 0; // Initialize j, the min vertex for upsampling convolution
        // Compute j(i), i.e., index in downsampled (j) from upsampled (i)
        for(IndexType d = 0 ; d < nx ; ++d){
            IndexType const vtd = vtds[d];
            IndexType const ndSource = nSource[d];
            IndexType const nlSourceProd = nlSourceProds[d];
            IndexType const vtlProd = vtlProds[d];
            j += ((IndexType)
                (((IndexType)(i/nlSourceProd)) % ndSource)*vtd/ndSource
            )*vtlProd;
        }
        IndexType jAux = 0;
        for(IndexType d = 0 ; d < nx ; ++d){
            IndexType const vtd = vtds[d];
            IndexType const vtlProd = vtlProds[d];
            IndexType const prodDownsampled = prodDownsampleds[d];
            jAux += (((IndexType)j/vtlProd)%vtd) * sUt * prodDownsampled;
        }
        j = jAux;
        // Register upsampling
        hUt.push_back(j);
    }
}


// ***  UTIL METHODS  *** //
// ********************** //
template <typename XDecimalType, typename IndexType>
IndexType
HierarchicalSparseGrid<XDecimalType, IndexType>::computeHalfPadding(
    IndexType const nx,
    arma::Col<IndexType> const &nt,
    IndexType const wt,
    IndexType const wDt,
    IndexType const sDt
){
    IndexType pt = 0; // Padding at depth t means finding max padding
    // Find padding needed for downsampling
    for(IndexType d = 0 ; d < nx ; ++d){
        // pDd : padding (p) Downsampling (D) depth (d) [hD, wD, sD]
        IndexType const pDdMod = (nt[d]-wDt) % sDt;
        IndexType const pDd = pDdMod == 0 ? 0 : sDt-pDdMod;
        if(pDd > pt) pt = pDd;
    }
    // Check whether submanifold padding is max padding
    IndexType const wt2 = 2*wt;
    if(wt2 > pt) pt = wt2;
    // Make sure the padding is evenly distributed (i.e., same at both sides)
    if(pt%2!=0) pt += 1;
    pt /= 2;
    // Return padding at depth t as max padding
    return pt;
}
