#include <pybind11/pybind11.h>
#include <module/DataMiningModule.hpp>
#include <module/ReceptiveFieldModule.hpp>
#include <module/AlgorithmModule.hpp>
#include <module/UtilModule.hpp>
#include <main/main_test.hpp>
#include <string>

#include <util/logging/BasicLogger.hpp>
#include <util/logging/GlobalLogger.hpp>


// ***  INITIALIZATION  *** //
// ************************ //
// Initialize C++ logging system
std::shared_ptr<BasicLogger> LOGGER = make_shared<BasicLogger>("VL3DPP.log");


// ***  DANGLING FUNCTIONS  *** //
// **************************** //
std::string get_hello_world() {
    return "HELLO WORLD";
}


// ***   PYVL3DPP MODULE   *** //
// *************************** //
PYBIND11_MODULE(pyvl3dpp, m){
    // Hello world
    m.def(
        "get_hello_world",
        &get_hello_world,
        "Return the HELLO WORLD string"
    );
    // TODO Rethink : Move main_test to vl3dpp_test ?
    m.def(
        "main_test",
        &vl3dpp::test::main_test,
        "Return the number of failed tests"
    );




    // ***  DATA MINING COMPONENTS  *** //
    // ******************************** //
    // Mine smooth features <XDecimalType, FDecimalType>
    m.def(
        "mine_smooth_feats_dd",                             // Function name
        &vl3dpp::pymods::mine_smooth_feats<double, double>, // Function wrapper
        "Mine smooth features"                              // Description
    );
    m.def(
        "mine_smooth_feats_df",
        &vl3dpp::pymods::mine_smooth_feats<double, float>,
        "Mine smooth features"
    );
    m.def(
        "mine_smooth_feats_fd",
        &vl3dpp::pymods::mine_smooth_feats<float, double>,
        "Mine smooth features"
    );
    m.def(
        "mine_smooth_feats_ff",
        &vl3dpp::pymods::mine_smooth_feats<float, float>,
        "Mine smooth features"
    );
    // Mine height features <XDecimalType, FDecimalType>
    m.def(
        "mine_height_feats_dd",                               // Name
        &vl3dpp::pymods::mine_height_feats<double, double>,   // Wrapper
        "Mine height features"                                // Descrip.
    );
    m.def(
        "mine_height_feats_df",
        &vl3dpp::pymods::mine_height_feats<double, float>,
        "Mine height features"
    );
    m.def(
        "mine_height_feats_fd",
        &vl3dpp::pymods::mine_height_feats<float, double>,
        "Mine height features"
    );
    m.def(
        "mine_height_feats_ff",
        &vl3dpp::pymods::mine_height_feats<float, float>,
        "Mine height features"
    );
    // Recount-based features <XDecimalType, FDecimalType>
    m.def(
        "mine_recount_dd",                              // Function name
        &vl3dpp::pymods::mine_recount<double, double>,  // Function wrapper
        "Mine recount-based features"                   // Description
    );
    m.def(
        "mine_recount_df",
        &vl3dpp::pymods::mine_recount<double, float>,
        "Mine recount-based features"
    );
    m.def(
        "mine_recount_fd",
        &vl3dpp::pymods::mine_recount<float, double>,
        "Mine recount-based features"
    );
    m.def(
        "mine_recount_ff",
        &vl3dpp::pymods::mine_recount<float, float>,
        "Mine recount-based features"
    );




    // ***  RECEPTIVE FIELD COMPONENTS  *** //
    // ************************************ //
    // Receptive field propagations
    m.def(
        "rf_propagate_mean_f8",
        &vl3dpp::pymods::rf_propagate_mean<float, arma::u8>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_f16",
        &vl3dpp::pymods::rf_propagate_mean<float, arma::u16>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_f32",
        &vl3dpp::pymods::rf_propagate_mean<float, arma::u32>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_f64",
        &vl3dpp::pymods::rf_propagate_mean<float, arma::u64>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_d8",
        &vl3dpp::pymods::rf_propagate_mean<double, arma::u8>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_d16",
        &vl3dpp::pymods::rf_propagate_mean<double, arma::u16>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_d32",
        &vl3dpp::pymods::rf_propagate_mean<double, arma::u32>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_d64",
        &vl3dpp::pymods::rf_propagate_mean<double, arma::u64>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_closest_f8",
        &vl3dpp::pymods::rf_propagate_closest<float, arma::u8>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_f16",
        &vl3dpp::pymods::rf_propagate_closest<float, arma::u16>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_f32",
        &vl3dpp::pymods::rf_propagate_closest<float, arma::u32>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_f64",
        &vl3dpp::pymods::rf_propagate_closest<float, arma::u64>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_d8",
        &vl3dpp::pymods::rf_propagate_closest<double, arma::u8>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_d16",
        &vl3dpp::pymods::rf_propagate_closest<double, arma::u16>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_d32",
        &vl3dpp::pymods::rf_propagate_closest<double, arma::u32>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_d64",
        &vl3dpp::pymods::rf_propagate_closest<double, arma::u64>,
        "Propagate closest value."
    );
    // Receptive field reductions
    m.def(
        "rf_reduce_mean_f8",
        &vl3dpp::pymods::rf_reduce_mean<float, arma::u8>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_f16",
        &vl3dpp::pymods::rf_reduce_mean<float, arma::u16>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_f32",
        &vl3dpp::pymods::rf_reduce_mean<float, arma::u32>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_f64",
        &vl3dpp::pymods::rf_reduce_mean<float, arma::u64>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_d8",
        &vl3dpp::pymods::rf_reduce_mean<double, arma::u8>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_d16",
        &vl3dpp::pymods::rf_reduce_mean<double, arma::u16>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_d32",
        &vl3dpp::pymods::rf_reduce_mean<double, arma::u32>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_d64",
        &vl3dpp::pymods::rf_reduce_mean<double, arma::u64>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_label_mode_u8u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    // Deep learning's receptive field pre-processing
    m.def(
        "rf_dl_fps_preproc_Xdf_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_fps_preproc<
            double, float, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_fps_preproc_Xff_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_fps_preproc<
            float, float, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_fps_preproc_Xdd_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_fps_preproc<
            double, double, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_fps_preproc_Xfd_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_fps_preproc<
            float, double, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_hfps_preproc_Xdf_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_hfps_preproc<
            double, float, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_hfps_preproc_Xff_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_hfps_preproc<
            float, float, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_hfps_preproc_Xdd_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_hfps_preproc<
            double, double, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_hfps_preproc_Xfd_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_hfps_preproc<
            float, double, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    // Deep learning's receptive field post-processing
    m.def(
        "rf_dl_fps_postproc_mean_fu8u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u8, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu8u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u8, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu8u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u8, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu8u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u8, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu16u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u16, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu16u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u16, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu16u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u16, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu16u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u16, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu32u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u32, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu32u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u32, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu32u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u32, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu32u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u32, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu64u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u64, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu64u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u64, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu64u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u64, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu64u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u64, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du8u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u8, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du8u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u8, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du8u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u8, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du8u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u8, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du16u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u16, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du16u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u16, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du16u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u16, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du16u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u16, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du32u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u32, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du32u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u32, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du32u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u32, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du32u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u32, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du64u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u64, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du64u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u64, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du64u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u64, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du64u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u64, arma::u64>,
        "Deep learning FPS post-processing."
    );




    // ***  ALGORITHM COMPONENTS  *** //
    // ****************************** //
    // Support neighborhoods
    m.def(
        "alg_support_neighborhoods_fs32u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::s32, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_ds32u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::s32, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fu32u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::u32, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_du32u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::u32, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fs16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::s16, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_ds16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::s16, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fu16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::u16, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_du16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::u16, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fs16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::s8, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_ds16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::s8, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fu16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::u8, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_du16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::u8, arma::u32>,
        "Compute support neighborhoods."
    );
    // Oversampler
    m.def(
        "alg_oversampler_du32",
        &vl3dpp::pymods::alg_oversampler<double, arma::u32>,
        "Compute oversampling."
    );
    m.def(
        "alg_oversampler_fu32",
        &vl3dpp::pymods::alg_oversampler<float, arma::u32>,
        "Compute oversampling."
    );



    // ***  LOGGING COMPONENTS  *** //
    // **************************** //
    m.def(
        "logging_enable",
        &vl3dpp::pymods::logging_enable,
        "Enable C++ logging."
    );
    m.def(
        "logging_disable",
        &vl3dpp::pymods::logging_disable,
        "Disable C++ logging."
    );
}