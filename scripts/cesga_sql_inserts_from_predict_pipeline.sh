#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPTS=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_X_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_X_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_X_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_X_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_X_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIr_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIr_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIr_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIr_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIr_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIrRGB_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIrRGB_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIrRGB_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIrRGB_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XIrRGB_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XRGB_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XRGB_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XRGB_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XRGB_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_kpc_final_XRGB_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_X_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_X_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_X_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_X_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_X_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIr_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIr_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIr_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIr_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIr_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIrRGB_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIrRGB_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIrRGB_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIrRGB_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XIrRGB_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XRGB_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XRGB_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XRGB_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XRGB_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnetpp_final_XRGB_buildveg_T5.sql'
)

# Model ID (from database) as environment variable
MODEL_IDS=(
    122 123 124 125 126
    127 128 129 130 131
    132 133 134 135 136
    137 138 139 140 141
    142 143 144 145 146
    147 148 149 150 151
    152 153 154 155 156
    157 158 159 160 161
)

# Paths to directories with the predictions
TRAINING_DIRS=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_buildveg/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_buildveg/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_buildveg/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_buildveg/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_buildveg/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_buildveg/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_buildveg/T5/preds/MERGE_68/'
)

# Dataset names
DATASET_NAMES=(
    'RGBIr_MERGE_21_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_216_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_260_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_68_minmaxnorm_BUILD_VEG'
)



# ---   M A I N   --- #
# ------------------- #
# Remove previous script, if any
rm -f ${OUTPUT_SCRIPT}

# Loop over training processes and merge into new script file
for (( i=0 ; i < ${#MODEL_IDS[@]} ; ++i )); do
    # Configure env. var. MODEL_ID
    export MODEL_ID=${MODEL_IDS[$i]}
    # Iterate datasets
    for (( j=0 ; j < ${#DATASET_NAMES[@]} ; ++j )); do
        k=$(( $i * ${#DATASET_NAMES[@]} ))
        python3 sql_insert_from_experiment.py "${TRAINING_DIRS[$k]}" "${DATASET_NAMES[$j]}" >> ${OUTPUT_SCRIPTS[$i]} &
    done
    # Join background jobs
    wait
    # Zip the SQL script
    zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPTS[$i]}) ${OUTPUT_SCRIPTS[$i]}
done


