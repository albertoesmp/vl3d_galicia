#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPTS=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_lmhveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_lmhveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_lmhveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_lmhveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_lmhveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_lmhveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_lmhveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_lmhveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_lmhveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_lmhveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_lmhveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_lmhveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_lmhveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_lmhveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_lmhveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_lmhveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_lmhveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_lmhveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_lmhveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_lmhveg_T5.sql'
)

# Model ID (from database) as environment variable
MODEL_IDS=(
    242 243 244 245 246
    247 248 249 250 251
    252 253 254 255 256
    257 258 259 260 261
)

# Paths to directories with the predictions
TRAINING_DIRS=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T1/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T1/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T1/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T1/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T2/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T2/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T2/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T2/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T3/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T3/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T3/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T3/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T4/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T4/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T4/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T4/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T5/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T5/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T5/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_lmhveg_alt/T5/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T1/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T1/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T1/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T1/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T2/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T2/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T2/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T2/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T3/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T3/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T3/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T3/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T4/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T4/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T4/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T4/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T5/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T5/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T5/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_lmhveg_alt/T5/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T1/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T1/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T1/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T1/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T2/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T2/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T2/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T2/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T3/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T3/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T3/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T3/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T4/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T4/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T4/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T4/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T5/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T5/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T5/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_lmhveg_alt/T5/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T1/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T1/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T1/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T1/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T2/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T2/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T2/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T2/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T3/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T3/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T3/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T3/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T4/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T4/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T4/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T4/preds/MERGE_189/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T5/preds/MERGE_234/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T5/preds/MERGE_97/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T5/preds/MERGE_239/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_lmhveg_alt/T5/preds/MERGE_189/'
)

# Dataset names
DATASET_NAMES=(
    'RGBIr_MERGE_234_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_97_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_239_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_189_minmaxnorm_BUILD_VEG'
)



# ---   M A I N   --- #
# ------------------- #
# Loop over training processes and merge into new script file
for (( i=0 ; i < ${#MODEL_IDS[@]} ; ++i )); do
    # Configure env. var. MODEL_ID
    export MODEL_ID=${MODEL_IDS[$i]}
    # Remove previous script, if any
    rm -f ${OUTPUT_SCRIPT[$i]}
    # Iterate datasets
    for (( j=0 ; j < ${#DATASET_NAMES[@]} ; ++j )); do
        k=$(( $i * ${#DATASET_NAMES[@]} ))
        #python3 sql_insert_from_experiment.py "${TRAINING_DIRS[$k]}" "${DATASET_NAMES[$j]}" >> ${OUTPUT_SCRIPTS[$i]} &  # TODO Restore : Legacy
        # TODO Remove : Alternative below
        python3 sql_insert_from_experiment.py "${TRAINING_DIRS[$k]}" "${DATASET_NAMES[$j]}" >> ${OUTPUT_SCRIPTS[$i]}
    done
    # Join background jobs
    wait
    # Zip the SQL script
    zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPTS[$i]}) ${OUTPUT_SCRIPTS[$i]}
done


