#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPTS=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_X_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIr_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XRGB_buildveg_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_buildveg_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_buildveg_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_buildveg_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_buildveg_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_pnet_final_XIrRGB_buildveg_T5.sql'
)

# Model ID (from database) as environment variable
MODEL_IDS=(
    302 303 304 305 306
    307 308 309 310 311
    312 313 314 315 316
    317 318 319 320 321
)

# Paths to directories with the predictions
TRAINING_DIRS=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T5/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T1/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T1/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T1/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T1/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T2/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T2/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T2/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T2/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T3/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T3/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T3/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T3/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T4/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T4/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T4/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T4/preds/MERGE_21/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T5/preds/MERGE_260/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T5/preds/MERGE_68/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T5/preds/MERGE_216/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T5/preds/MERGE_21/'
)

# Dataset names
DATASET_NAMES=(
    'RGBIr_MERGE_260_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_68_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_216_minmaxnorm_BUILD_VEG'
    'RGBIr_MERGE_21_minmaxnorm_BUILD_VEG'
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


