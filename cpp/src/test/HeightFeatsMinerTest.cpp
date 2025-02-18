#include <test/HeightFeatsMinerTest.hpp>
#include <util/VL3DPPException.hpp>

#include <sstream>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace vl3dpp::test;
using vl3dpp::util::VL3DPPException;
using vl3dpp::mining::HeightFeatsMiner;


// ***   CONSTRUCTION / DESTRUCTION  *** //
// ************************************* //
HeightFeatsMinerTest::HeightFeatsMinerTest(float const eps) :
    BaseTest("Height features miner test"),
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
            ss  << "HeightFeaturesMinerTest failed to read test data from "
                << "\"" << Ppath << "\".";
            throw VL3DPPException(ss.str());
        }
    }
    catch(std::exception &ex){
        correctlyConstructed = false;
        std::cout   << "HeightFeaturesMinerTest could NOT be built."
                    << std::endl;
        std::cout   << "HeightFeaturesMinerTest exception: "
                    << ex.what() << std::endl;
    }

    // Populate test data matrices (X, Fref) from loaded data (P)
    X = P.cols(0, 2);
    Fref = arma::reverse(P.cols(3, 12), 1);
    // Center test structure space
    arma::Row<float> xmin = arma::min(X, 0);
    arma::Row<float> xmax = arma::max(X, 0);
    X.each_row() -= ((xmin+xmax)/2.0);
}


// ***   R U N   *** //
// ***************** //
bool HeightFeatsMinerTest::run(){
    // Compute height features
    HeightFeatsMiner<float, float> hfm(
        "cylinder",
        5.0,
        0,
        "",
        std::vector<std::string>({
            "floor_coordinate", "floor_distance",
            "ceil_coordinate", "ceil_distance",
            "mean_height", "height_stdev", "height_variance",
            "height_skewness", "height_kurtosis","height_range"
        }),
        -1
    );
    arma::Mat<float> Fhat = hfm.mine(X);
    // Validate computed height features
    if(!arma::approx_equal(Fhat, Fref, "both", eps, eps)) return false;
    // If this point is reached, then all test passed
    return true;
}
