#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPTS=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_X_fullraw_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_X_fullraw_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_X_fullraw_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_X_fullraw_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_X_fullraw_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIr_fullraw_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIr_fullraw_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIr_fullraw_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIr_fullraw_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIr_fullraw_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XRGB_fullraw_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XRGB_fullraw_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XRGB_fullraw_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XRGB_fullraw_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XRGB_fullraw_T5.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIrRGB_fullraw_T1.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIrRGB_fullraw_T2.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIrRGB_fullraw_T3.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIrRGB_fullraw_T4.sql'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts_sflnet_try1_XIrRGB_fullraw_T5.sql'
)

# Model ID (from database) as environment variable
MODEL_IDS=(
    322 323 324 325 326
    327 328 329 330 331
    332 333 334 335 336
    337 338 339 340 341
)

# Paths to directories with the predictions
TRAINING_DIRS=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T1/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T1/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T1/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T1/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T2/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T2/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T2/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T2/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T3/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T3/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T3/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T3/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T4/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T4/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T4/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T4/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T5/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T5/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T5/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T5/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T1/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T1/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T1/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T1/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T2/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T2/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T2/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T2/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T3/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T3/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T3/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T3/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T4/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T4/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T4/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T4/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T5/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T5/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T5/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T5/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T1/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T1/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T1/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T1/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T2/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T2/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T2/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T2/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T3/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T3/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T3/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T3/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T4/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T4/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T4/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T4/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T5/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T5/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T5/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T5/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T1/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T1/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T1/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T1/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T2/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T2/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T2/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T2/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T3/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T3/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T3/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T3/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T4/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T4/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T4/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T4/preds/MERGE_13/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T5/preds/MERGE_195/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T5/preds/MERGE_212/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T5/preds/MERGE_108/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T5/preds/MERGE_13/'
)

# Dataset names
DATASET_NAMES=(
    'RGBIr_MERGE_195_minmaxnorm_FULLRAW'
    'RGBIr_MERGE_212_minmaxnorm_FULLRAW'
    'RGBIr_MERGE_108_minmaxnorm_FULLRAW'
    'RGBIr_MERGE_13_minmaxnorm_FULLRAW'
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
        k=$(( $i * ${#DATASET_NAMES[@]} + $j ))
        #python3 sql_insert_from_experiment.py "${TRAINING_DIRS[$k]}" "${DATASET_NAMES[$j]}" >> ${OUTPUT_SCRIPTS[$i]} &  # TODO Restore : Legacy
        # TODO Remove : Alternative below
        python3 sql_insert_from_experiment.py "${TRAINING_DIRS[$k]}" "${DATASET_NAMES[$j]}" >> ${OUTPUT_SCRIPTS[$i]}
    done
    # Join background jobs
    wait
    # Zip the SQL script
    zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPTS[$i]}) ${OUTPUT_SCRIPTS[$i]}
done


