#include <test/SmoothFeatsMinerTest.hpp>
#include <util/VL3DPPException.hpp>

#include <sstream>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace vl3dpp::test;
using vl3dpp::util::VL3DPPException;
using vl3dpp::mining::SmoothFeatsMiner;


// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
SmoothFeatsMinerTest::SmoothFeatsMinerTest(float const eps) :
    BaseTest("Smooth features miner test"),
    eps(eps)
{
    // Test data (P) path
    std::stringstream ss;
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << "DataMining_XF.xyz";
    std::string Ppath = ss.str();

    // Read test data (P)
    arma::Mat<float> P;
    try{
        if(!P.load(Ppath, arma::file_type::csv_ascii)){
            ss.str("");
            ss  << "SmoothFeaturesMinerTest failed to read test data from "
                << "\"" << Ppath << "\".";
            throw VL3DPPException(ss.str());
        }
    }
    catch(std::exception &ex){
        correctlyConstructed = false;
        std::cout   << "SmoothFeaturesMinerTest could NOT be built."
                    << std::endl;
        std::cout   << "SmoothFeaturesMinerTest exception: "
                    << ex.what() << std::endl;
    }

    // Populate test data matrices (X, F, Fref) from loaded data (P)
    X = P.cols(0, 2);
    Fref = arma::reverse(P.cols(13, 18), 1);
    F = arma::reverse(P.cols(19, 20), 1);
    // Center test structure space
    arma::Row<float> xmin = arma::min(X, 0);
    arma::Row<float> xmax = arma::max(X, 0);
    X.each_row() -= ((xmin+xmax)/2.0);
}


// ***   R U N   *** //
// ***************** //
bool SmoothFeatsMinerTest::run(){
    // Compute smooth features
    std::vector<std::string> fnames({"mean", "weighted_mean", "gaussian_rbf"});
    SmoothFeatsMiner<float, float> sfm(
        "sphere",
        5.0,
        arma::Col<float>({5.0, 5.0, 5.0}),
        0,
        0,
        5.0,
        1.0,
        "replace",
        fnames,
        -1
    );
    arma::Mat<float> Fhat = sfm.mine(X, F);
    // Validate computed smooth features
    if(!arma::approx_equal(Fhat, Fref, "both", eps, eps)) return false;
    // If this point is reached, then all tests passed
    return true;
}
