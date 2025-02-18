#include <test/RecountMinerTest.hpp>
#include <util/VL3DPPException.hpp>

#include <sstream>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace vl3dpp::test;
using vl3dpp::util::VL3DPPException;
using vl3dpp::mining::RecountMiner;


// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
RecountMinerTest::RecountMinerTest(float const eps) :
    BaseTest("Recount miner test"),
    eps(eps)
{
    // Test data (P)
    std::stringstream ss;
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << "DataMining_XF.xyz";
    std::string Ppath = ss.str();

    // Read test data (P)
    arma::Mat<double> P;
    try{
        if(!P.load(Ppath, arma::file_type::csv_ascii)){
            ss.str("");
            ss  << "RecountMinerTest failed to read test data from "
                << "\"" << Ppath << "\".";
            throw VL3DPPException(ss.str());
        }
    }
    catch(std::exception &ex){
        correctlyConstructed = false;
        std::cout   << "RecountMinerTest could NOT be built."
                    << std::endl;
        std::cout   << "RecountMinerTest exception: "
                    << ex.what() << std::endl;
    }

    // Center test structure space
    arma::Mat<double> _X = P.cols(0, 2);
    arma::Row<double> xmin = arma::min(_X, 0);
    arma::Row<double> xmax = arma::max(_X, 0);
    _X.each_row() -= ((xmin+xmax)/2.0);
    // Populate test data matrices (X, F) from loaded data (P)
    X = arma::conv_to<arma::Mat<float>>::from(_X);
    F = arma::conv_to<arma::Mat<float>>::from(P.cols(19, 20));

    // Test reference data (Fref)
    ss.str("");
    ss  << "test_data" << std::filesystem::path::preferred_separator
        << "DataMining_Recounts.xyz";
    std::string FrefPath = ss.str();

    // Read test data (Fref)
    try{
        if(!Fref.load(FrefPath, arma::file_type::csv_ascii)){
            ss.str("");
            ss  << "RecountMinerTest failed to read reference test deta from "
                << "\"" << FrefPath << "\".";
            throw VL3DPPException(ss.str());
        }
    }
    catch(std::exception &ex){
        correctlyConstructed = false;
        std::cout   << "RecountMinerTest could NOT be built."
                    << std::endl;
        std::cout   << "RecountMinerTest exception: "
                    << ex.what() << std::endl;
    }
}

// ***   R U N   *** //
// ***************** //
bool RecountMinerTest::run(){
    // Compute recount features
    RecountMiner<float, float> rm(
        "cylinder", // Neighborhood type
        0, // Neighborhood k (for knn)
        arma::Col<float>({5.0, 0.0, 0.0}), // Neighborhood radius
        0, // Neighborhood lower bound
        0, // Neighborhood upper bound
        std::vector<bool>({false, true, true}), // Ignore NaN
        std::vector<bool>({true, true, true}), // Absolute frequency
        std::vector<bool>({true, true, true}), // Relative frequency
        std::vector<bool>({true, true, true}), // Surface density
        std::vector<bool>({true, true, true}), // Volume density
        std::vector<int>({32, 32, 32}), // Vertical segments
        std::vector<int>({0, 8, 8}), // Rings
        std::vector<int>({0, 8, 8}), // Radial boundaries
        std::vector<int>({0, 16, 16}), // Sectors 2D
        std::vector<int>({0, 32, 32}), // Sectors 3D
        std::vector<std::vector<arma::uword>>({ // Value indices
            std::vector<arma::uword>({}),
            std::vector<arma::uword>({0}),
            std::vector<arma::uword>({1})
        }),
        std::vector<std::vector<std::string>>({ // Condition types
            std::vector<std::string>({}),
            std::vector<std::string>({"less_than_or_equal_to"}),
            std::vector<std::string>({"greater_than_or_equal_to"})
        }),
        std::vector<std::vector<std::vector<float>>>({ // Value target
            std::vector<std::vector<float>>({}),
            std::vector<std::vector<float>>({std::vector<float>({0.5})}),
            std::vector<std::vector<float>>({std::vector<float>({0.5})})
        }),
        -1
    );
    arma::Mat<float> Fhat = rm.mine(X, F);
    // Replace max values to use the same max constant
    float const refMax = Fref.col(3).max();
    float const maxTh = Fref.n_rows;
    for(arma::uword i = 0 ; i < Fhat.n_rows ; ++i){
        if(Fhat.at(i, 3) > maxTh) Fhat.at(i, 3) = refMax;
        if(Fhat.at(i, 8) > maxTh) Fhat.at(i, 8) = refMax;
        if(Fhat.at(i, 17) > maxTh) Fhat.at(i, 17) = refMax;
    }
    // Validate computed smooth features
    if(!arma::approx_equal(Fhat, Fref, "both", eps, eps)) return false;
    // If this point is reached, then all tests passed
    return true;
}