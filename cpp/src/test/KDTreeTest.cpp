#include <test/KDTreeTest.hpp>
#include <test/OctreeTest.hpp>
#include <util/VL3DPPException.hpp>
#include <sstream>
#include <filesystem>
#include <memory>

using namespace vl3dpp::test;
using vl3dpp::util::VL3DPPException;

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
KDTreeTest::KDTreeTest(double const eps) :
    BaseTest("KDTree test"),
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
            ss  << "KDTreeTest failed to read test data from "
                << "\"" << Xpath << "\".";
            throw VL3DPPException(ss.str());
        }
        if(!Xsup.load(XsupPath, arma::file_type::csv_ascii)){
            ss.str("");
            ss  << "KDtreeTest failed to read test data from "
                << "\"" << XsupPath << "\".";
            throw VL3DPPException(ss.str());
        }
    }
    catch (std::exception &ex) {
        correctlyConstructed = false;
        std::cout << "KDTreeTest could NOT be built." << std::endl;
        std::cout << "KDTreeTest exception: " << ex.what() << std::endl;
    }

    // Build KDTree
    kdt = std::make_shared<KDTree<size_t, double>>(X, true, true);
}

// ***   R U N   *** //
// ***************** //
bool KDTreeTest::run(){
    // Extract view of z coordinate
    arma::subview_col<double> const &z = X.col(2);
    // 3D KNN test
    size_t const k = 16;
    arma::Mat<size_t> N(Xsup.n_rows, k);
    arma::Mat<double> D(Xsup.n_rows, k);
    kdt->findKnn(Xsup, k, N, D);
    if(!validateNeighborhoods(N, "Octree_knn3D_N.txt")) return false;
    if(!validateDistances(D, "Octree_knn3D_D.txt")) return false;
    // 2D KNN test
    kdt->findKnn2D(Xsup, k, N, D);
    if(!validateNeighborhoods(N, "Octree_knn2D_N.txt", true)) return false;
    if(!validateDistances(D, "Octree_knn2D_D.txt")) return false;
    // Bounded KNN test
    double const maxDistanceBound = 0.2*0.2;
    std::vector<arma::Col<size_t>> vecN;
    std::vector<arma::Col<double>> vecD;
    kdt->findBoundedKnn(Xsup, k, maxDistanceBound, vecN, vecD);
    if(!validateNeighborhoods(vecN, "Octree_boundedKnn3D_N.txt", false, true))
        return false;
    if(!validateDistances(vecD, "Octree_boundedKnn3D_D.txt", true))
        return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Bounded 2D KNN test
    kdt->findBoundedKnn2D(Xsup, k, maxDistanceBound, vecN, vecD);
    if(!validateNeighborhoods(vecN, "Octree_boundedKnn2D_N.txt", true, true))
        return false;
    if(!validateDistances(vecD, "Octree_boundedKnn2D_D.txt", true))
        return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Spherical test
    double const radius = 0.3;
    kdt->findSphere(Xsup, radius, vecN, vecD);
    if(!validateNeighborhoods(
        vecN, "Octree_sphere3D_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Cylindrical test
    kdt->findCylinder(Xsup, radius, vecN, vecD);
    if(!validateNeighborhoods(
        vecN, "Octree_sphere2D_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Bounded cylindrical test
    double const zmin = -1.01, zmax = 1.01;
    kdt->findBoundedCylinder(Xsup, z, radius, zmin, zmax, vecN, vecD);
    if(!validateNeighborhoods(
        vecN, "Octree_boundedCylinder_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Box test
    arma::Col<double> halfLength({0.3, 0.3, 0.3});
    kdt->findBox(Xsup, halfLength, vecN);
    if(!validateNeighborhoods(
        vecN, "Octree_rectangular3D_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Rectangle test
    kdt->findRectangle(Xsup, halfLength, vecN);
    if(!validateNeighborhoods(
        vecN, "Octree_rectangular2D_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Bounded rectangle test
    kdt->findBoundedRectangle(Xsup, z, halfLength, zmin, zmax, vecN);
    if(!validateNeighborhoods(
        vecN, "Octree_boundedRectangular_N.txt", false, false, true
    )) return false;
    vecN.clear();   vecD.clear(); // Clear neighborhood data
    // Test memory operations
    KDTree<size_t, double> kdtA (X, true, true);
    KDTree<size_t, double> kdtB (X, true, true);
    void *kdt3DB = kdtB.kdt3D;
    void *kdt2DB = kdtB.kdt2D;
    kdtA = std::move(kdtB);
    if(kdtA.kdt3D != kdt3DB) return false;
    if(kdtA.kdt2D != kdt2DB) return false;
    if(kdtB.kdt3D == kdt3DB) return false;
    if(kdtB.kdt2D == kdt2DB) return false;
    // If this point is reached, then all test passed
    return true;
}




// ***  UTIL METHODS  *** //
// ********************** //
bool KDTreeTest::validateNeighborhoods(
    arma::Mat<size_t> const &N,
    std::string const &NPath,
    bool const force2D,
    bool const sort
){
    return OctreeTest::_validateNeighborhoods(
        X, Xsup, N, NPath, force2D, sort, eps
    );
}

bool KDTreeTest::validateNeighborhoods(
    std::vector<arma::Col<size_t>> const &N,
    std::string const &NPath,
    bool const force2D,
    bool const bounded,
    bool const sort
){
    return OctreeTest::_validateNeighborhoods(
        X, Xsup, N, NPath, force2D, bounded, sort, eps
    );
}

bool KDTreeTest::validateDistances(
    arma::Mat<double> const &D,
    std::string const &DPath
){
    return OctreeTest::_validateDistances(D, DPath, eps);
}

bool KDTreeTest::validateDistances(
    std::vector<arma::Col<double>> const &D,
    std::string const &DPath,
    bool const bounded
){
    return OctreeTest::_validateDistances(D, DPath, bounded, eps);
}
