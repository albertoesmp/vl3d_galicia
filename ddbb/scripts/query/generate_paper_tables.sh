# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
#
# BRIEF: Script to generate LaTeX tables from SQL queries for the VL3D-Galicia
#   paper.
#
# HINT: Launch as show below:
# PGPASSWORD=<DATABASE PASSWORD HERE> ./generate_paper_tables.sh
# ----------------------------------------------------------------------------


# ---   CONSTANTS   --- #
# --------------------- #
IFS="$"
HOST='172.17.0.3'
PORT='5432'
REGION=(
    'A Coruña'
    'Lugo'
    'Pontevedra'
    'Ourense'
    'Galicia West'
    'Galicia East'
    'Galicia'
)


# ---  TABLES : VEGETATION  --- #
# ----------------------------- #
# Global vegetation
echo 'GLOBAL VEGETATION TABLE:'
i=0;
for line in $(psql -h ${HOST} -p ${PORT} -U postgres -d catadb --csv -t -F '&' -f query_vegetation_global.sql | awk 'BEGIN {FS=","} {for(i=4; i <= NF ; ++i){printf("%.2f", $i); if(i==NF){printf("$");} else{printf(" & ");}}}'); do
    echo "${REGION[${i}]} & ${line} \\\\"
    i=$(( i + 1 ))
done
echo -e "-----------------------------------------------------------\n"


# ---  TABLES : LMH-VEGETATION  --- #
# --------------------------------- #
# Global LMH-Vegetation
echo 'GLOBAL LMH-VEGETATION TABLE:'
i=0;
for line in $(psql -h ${HOST} -p ${PORT} -U postgres -d catadb --csv -t -F '&' -f query_lmhveg_global.sql | awk 'BEGIN {FS=","} {for(i=4; i <= NF ; ++i){printf("%.2f", $i); if(i==NF){printf("$");} else{printf(" & ");}}}'); do
    echo "${REGION[${i}]} & ${line} \\\\"
    i=$(( i + 1 ))
done
echo -e "-----------------------------------------------------------\n"

# ---  TABLES : BUILDING  --- #
# --------------------------- #
# Global building
echo 'GLOBAL BUILDING TABLE:'
i=0;
for line in $(psql -h ${HOST} -p ${PORT} -U postgres -d catadb --csv -t -F '&' -f query_building_global.sql | awk 'BEGIN {FS=","} {for(i=4; i <= NF ; ++i){printf("%.2f", $i); if(i==NF){printf("$");} else{printf(" & ");}}}'); do
    echo "${REGION[${i}]} & ${line} \\\\"
    i=$(( i + 1 ))
done
echo -e "-----------------------------------------------------------\n"

# ---  TABLES : BUILDVEG  --- #
# --------------------------- #
# Global buildved
echo 'GLOBAL BUILDVEG TABLE:'
i=0;
for line in $(psql -h ${HOST} -p ${PORT} -U postgres -d catadb --csv -t -F '&' -f query_buildveg_global.sql | awk 'BEGIN {FS=","} {for(i=4; i <= NF ; ++i){printf("%.2f", $i); if(i==NF){printf("$");} else{printf(" & ");}}}'); do
    echo "${REGION[${i}]} & ${line} \\\\"
    i=$(( i + 1 ))
done
echo -e "-----------------------------------------------------------\n"

# ---  TABLES : FULLRAW  --- #
# -------------------------- #
# Global fullraw (random forest)
echo 'GLOBAL FULLRAW RANDOM FOREST TABLE:'
i=0;
for line in $(psql -h ${HOST} -p ${PORT} -U postgres -d catadb --csv -t -F '&' -f query_fullraw_rf_global.sql | awk 'BEGIN {FS=","} {for(i=4; i <= NF ; ++i){printf("%.2f", $i); if(i==NF){printf("$");} else{printf(" & ");}}}'); do
    echo "${REGION[${i}]} & ${line} \\\\"
    i=$(( i + 1 ))
done
echo -e "-----------------------------------------------------------\n"
# Global fullraw (random forest)
echo 'GLOBAL FULLRAW SFL-NET TABLE:'
i=0;
for line in $(psql -h ${HOST} -p ${PORT} -U postgres -d catadb --csv -t -F '&' -f query_fullraw_global.sql | awk 'BEGIN {FS=","} {for(i=4; i <= NF ; ++i){printf("%.2f", $i); if(i==NF){printf("$");} else{printf(" & ");}}}'); do
    echo "${REGION[${i}]} & ${line} \\\\"
    i=$(( i + 1 ))
done
echo -e "-----------------------------------------------------------\n"
