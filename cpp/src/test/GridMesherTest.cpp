#include <test/GridMesherTest.hpp>

using vl3dpp::test::GridMesherTest;

// ***   R U N   *** //
// ***************** //
bool GridMesherTest::run(){
    // Instantiate GridMesher
    GridMesher<double> gm(arma::Col<double>({0.1, 0.1, 0.1}), false);
    // Check 2D grid against ND grid
    arma::Col<double> const a2D({-1.0, -1.0});
    arma::Col<double> const b2D({1.0, 1.0});
    arma::Mat<double> const G2D = gm.computeNodes2D(a2D, b2D);
    arma::Mat<double> const GND2 = gm.computeNodesND(a2D, b2D);
    if(!checkGridEquality(G2D, GND2)) return false;
    // Check 3D grid against ND grid
    arma::Col<double> const a3D({-1.0, -1.0, -1.0});
    arma::Col<double> const b3D({1.0, 1.0, 1.0});
    arma::Mat<double> const G3D = gm.computeNodes3D(a3D, b3D);
    arma::Mat<double> const GND3 = gm.computeNodesND(a3D, b3D);
    if(!checkGridEquality(G3D, GND3)) return false;
    // On all tests passed
    return true;
}


// ***  UTIL METHODS  *** //
// ********************** //
template <typename DecimalType>
bool GridMesherTest::checkGridEquality(
    arma::Mat<DecimalType> const &Ga,
    arma::Mat<DecimalType> const &Gb
) const{
    // Check dimensionality of matrices
    if(Ga.n_rows != Gb.n_rows) return false;
    if(Ga.n_cols != Gb.n_cols) return false;
    // Check the node-wise coordinates
    if(arma::any(arma::vectorise(arma::abs(Ga-Gb) > eps))) return false;
    // All checks passed
    return true;
}
