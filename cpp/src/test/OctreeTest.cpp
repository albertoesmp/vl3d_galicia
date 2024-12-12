#include <test/OctreeTest.hpp>
#include <util/VL3DPPException.hpp>
#include <sstream>
#include <filesystem>
#include <memory>

using namespace vl3dpp::test;
using vl3dpp::util::VL3DPPException;

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
OctreeTest::OctreeTest(double const eps) :
    BaseTest("Octree test"),
    eps(eps)
{
    // Test data (X, Xsup) path
    std::stringstream ss;
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << "Octree_X.xyz";
    std::string const Xpath = ss.str();
    ss.str("");
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << "Octree_Xsup.xyz";
    std::string const XsupPath = ss.str();

    // Read test data
    try {
        if(!X.load(Xpath, arma::file_type::csv_ascii)){
            ss.str("");
            ss  << "OctreeTest failed to read test data from "
                << "\"" << Xpath << "\".";
            throw VL3DPPException(ss.str());
        }
        if(!Xsup.load(XsupPath, arma::file_type::csv_ascii)){
            ss.str("");
            ss  << "OctreeTest failed to read test data from "
                << "\"" << XsupPath << "\".";
            throw VL3DPPException(ss.str());
        }
    }
    catch (std::exception &ex) {
        correctlyConstructed = false;
        std::cout << "OctreeTest could NOT be built." << std::endl;
        std::cout << "OctreeTest exception: " << ex.what() << std::endl;
    }

    // Build octree
    octree = std::make_shared<Octree<size_t, double>>(X, true, true, 1.0);
}




// ***  R U N  *** //
// *************** //
bool OctreeTest::run(){
    // Extract view of z coordinate
    arma::subview_col<double> const &z = X.col(2);
    // 3D KNN test
    size_t const k = 16;
    arma::Mat<size_t> N(Xsup.n_rows, k);
    arma::Mat<double> D(Xsup.n_rows, k);
    octree->findKnn(Xsup, k, N, D);
    if(!validateNeighborhoods(N, "Octree_knn3D_N.txt")) return false;
    if(!validateDistances(D, "Octree_knn3D_D.txt")) return false;
    // 2D KNN test
    octree->findKnn2D(Xsup, k, N, D);
    if(!validateNeighborhoods(N, "Octree_knn2D_N.txt", true)) return false;
    if(!validateDistances(D, "Octree_knn2D_D.txt")) return false;
    // Bounded KNN test
    double const maxDistanceBound = 0.2*0.2;
    std::vector<arma::Col<size_t>> vecN;
    std::vector<arma::Col<double>> vecD;
    octree->findBoundedKnn(Xsup, k, maxDistanceBound, vecN, vecD);
    if(!validateNeighborhoods(vecN, "Octree_boundedKnn3D_N.txt", false, true))
        return false;
    if(!validateDistances(vecD, "Octree_boundedKnn3D_D.txt", true))
        return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Bounded 2D KNN test
    octree->findBoundedKnn2D(Xsup, k, maxDistanceBound, vecN, vecD);
    if(!validateNeighborhoods(vecN, "Octree_boundedKnn2D_N.txt", true, true))
        return false;
    if(!validateDistances(vecD, "Octree_boundedKnn2D_D.txt", true))
        return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Spherical test
    double const radius = 0.3;
    octree->findSphere(Xsup, radius, vecN, vecD);
    if(!validateNeighborhoods(
        vecN, "Octree_sphere3D_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Cylindrical test
    octree->findCylinder(Xsup, radius, vecN, vecD);
    if(!validateNeighborhoods(
        vecN, "Octree_sphere2D_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Bounded cylindrical test
    double const zmin = -1.01, zmax = 1.01;
    octree->findBoundedCylinder(Xsup, z, radius, zmin, zmax, vecN, vecD);
    if(!validateNeighborhoods(
        vecN, "Octree_boundedCylinder_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Box test
    arma::Col<double> halfLength({0.3, 0.3, 0.3});
    octree->findBox(Xsup, halfLength, vecN);
    if(!validateNeighborhoods(
        vecN, "Octree_rectangular3D_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Rectangle test
    octree->findRectangle(Xsup, halfLength, vecN);
    if(!validateNeighborhoods(
        vecN, "Octree_rectangular2D_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Bounded rectangle test
    octree->findBoundedRectangle(Xsup, z, halfLength, zmin, zmax, vecN);
    if(!validateNeighborhoods(
        vecN, "Octree_boundedRectangular_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Test memory operations
    Octree<size_t, double> otA (X, true, true);
    Octree<size_t, double> otB (X, true, true);
    void *ot3DB = otB.octree3D;
    void *ot2DB = otB.octree2D;
    otA = std::move(otB);
    if(otA.octree3D != ot3DB) return false;
    if(otA.octree2D != ot2DB) return false;
    if(otB.octree3D == ot3DB) return false;
    if(otB.octree2D == ot2DB) return false;
    // If this point is reached, then all test passed
    return true;
}




// ***  UTIL METHODS  *** //
// ********************** //
bool OctreeTest::_validateNeighborhoods(
    arma::Mat<double> const &X,
    arma::Mat<double> const &Xsup,
    arma::Mat<size_t> const &N,
    std::string const &_NPath,
    bool const force2D,
    bool const sort,
    double const eps
){
    // Load reference neighborhood matrix
    std::stringstream ss;
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << _NPath;
    std::string const NPath = ss.str();
    arma::Mat<size_t> NRef;
    if(!NRef.load(NPath, arma::file_type::csv_ascii)){
        return false;
    }
    // Compare
    for(size_t i = 0 ; i < N.n_rows ; ++i){
        // Neighborhood
        arma::Col<size_t> ni = N.row(i).as_col();
        if(sort) ni = arma::sort(N.row(i).as_col());
        // Reference neighborhood
        arma::Col<size_t> niRef = NRef.row(i).as_col();
        if(sort) niRef = arma::sort(NRef.row(i).as_col());
        // Support point
        arma::Col<double> xsupi = Xsup.row(i).as_col();
        if(force2D) xsupi[2] = 0.0;
        for(size_t j = 0 ; j < ni.n_elem ; ++j) {
            // Neighbor point
            arma::Col<double> x = X.row(ni[j]).as_col();
            if(force2D) x[2] = 0.0;
            // Reference neighbor point
            arma::Col<double> xRef = X.row(niRef[j]).as_col();
            if(force2D) xRef[2] = 0.0;
            // Decimal check on the magnitudes/norms
            if(std::abs(arma::norm(x - xsupi)-arma::norm(xRef - xsupi)) > eps)
                return false;
        }
    }
    // All checks passed
    return true;
}

bool OctreeTest::validateNeighborhoods(
    arma::Mat<size_t> const &N,
    std::string const &NPath,
    bool const force2D,
    bool const sort
){
    return _validateNeighborhoods(X, Xsup, N, NPath, force2D, sort, eps);
}

bool OctreeTest::_validateNeighborhoods(
    arma::Mat<double> const &X,
    arma::Mat<double> const &Xsup,
    std::vector<arma::Col<size_t>> const &N,
    std::string const &_NPath,
    bool const force2D,
    bool const bounded,
    bool const sort,
    double const eps
){
    // Load reference neighborhoods
    std::stringstream ss;
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << _NPath;
    std::string const NPath = ss.str();
    std::vector<arma::Col<size_t>> NRef;
    try{
        NRef = BaseTest::readRaggedMatrix<size_t>(NPath);
    }
    catch(VL3DPPException &vl3dppex){
        return false;
    }
    // Compare
    for(size_t i = 0 ; i < N.size() ; ++i){
        // Neighborhood
        arma::Col<size_t> ni = sort ? arma::sort(N[i]) : N[i];
        // Reference neighborhood
        arma::Col<size_t> niRef = bounded ?
            NRef[i].rows(0, ni.n_elem-1) : NRef[i];
        if(sort) niRef = arma::sort(niRef);
        // Support point
        arma::Col<double> xsupi = Xsup.row(i).as_col();
        if(force2D) xsupi[2] = 0.0;
        for(size_t j = 0 ; j < ni.n_elem ; ++j){
            // Neighbor point
            arma::Col<double> x = X.row(ni[j]).as_col();
            if(force2D) x[2] = 0.0;
            // Reference neighbor point
            arma::Col<double> xRef = X.row(niRef[j]).as_col();
            if(force2D) xRef[2] = 0.0;
            // Decimal check on the magnitudes/norms
            if(std::abs(arma::norm(x - xsupi)-arma::norm(xRef - xsupi)) > eps)
                return false;
        }
    }
    // All checks passed
    return true;
}

bool OctreeTest::validateNeighborhoods(
    std::vector<arma::Col<size_t>> const &N,
    std::string const &NPath,
    bool const force2D,
    bool const bounded,
    bool const sort
){
    return _validateNeighborhoods(
        X, Xsup, N, NPath, force2D, bounded, sort, eps
    );
}

bool OctreeTest::_validateDistances(
    arma::Mat<double> const &D,
    std::string const &_DPath,
    double const eps
){
    // Load reference distance matrix
    std::stringstream ss;
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << _DPath;
    std::string const DPath = ss.str();
    arma::Mat<double> DRef;
    if(!DRef.load(DPath, arma::file_type::csv_ascii)){
        return false;
    }
    // Compare
    for(size_t i = 0 ; i < D.n_rows ; ++i){
        if(arma::any(arma::abs(D.row(i)-arma::square(DRef.row(i))) > eps))
            return false;
    }
    // All checks passed
    return true;
}

bool OctreeTest::validateDistances(
    arma::Mat<double> const &D,
    std::string const &DPath
){
    return _validateDistances(D, DPath, eps);
}

bool OctreeTest::_validateDistances(
    std::vector<arma::Col<double>> const &D,
    std::string const &_DPath,
    bool const bounded,
    double const eps
){
    // Load reference distances
    std::stringstream ss;
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << _DPath;
    std::string const DPath = ss.str();
    std::vector<arma::Col<double>> DRef;
    try{
        DRef = BaseTest::readRaggedMatrix<double>(DPath);
    }
    catch(VL3DPPException &vl3dppex){
        return false;
    }
    // Compare
    for(size_t i = 0 ; i < D.size() ; ++i){
        arma::Col<double> const di = D[i];
        arma::Col<double> const diRef = bounded ?
            DRef[i].rows(0, di.n_elem-1) : DRef[i];
        if(arma::any(arma::abs(di-arma::square(diRef)) > eps)) return false;
    }
    // All checks passed
    return true;
}

bool OctreeTest::validateDistances(
    std::vector<arma::Col<double>> const &D,
    std::string const &DPath,
    bool const bounded
){
    return _validateDistances(D, DPath, bounded, eps);
}

