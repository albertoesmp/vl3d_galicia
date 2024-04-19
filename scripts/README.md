# Ejecución clasificación PNOA2 en Galicia

1 - Definir las siguientes variables en `cesga/vl3d_cesga_env.sh`

    - VL3D_DIR: Ruta al directorio de este repositorio.
    - VL3D_SCRIPT: Ruta absoluta a `vl3d.py` de este repositorio.
    - VL3D_ENV: Ruta al entorno de conda que se va a usar para este proyecto.

    =================================================
    /!\ NO HACER SOURCE DE `cesga/vl3d_cesga_env.sh` /!\
    =================================================

    Exportar las 3 variables al entorno antes de seguir con los siguientes pasos. 


2 - Definir las siguientes variables en `scripts/experiment_generation/replace_paths.sh`:

    - MODEL: Ruta al modelo del experimento que se va a ejecutar (ver final del documento)
    - EXPERIMENT_NAME: Nombre del experimento
    - OUTPUT_PATH: Ruta donde se guardarán los resultados. Debe estar en `$STORE2`.

3 - Ejecutar el script pasando con -i el directorio (`experiments/nombre_experimento/nombre_usuario`) donde están todos los .json generados que se van a ejecutar.

4 - `chmod +x scripts/slurm/work.sh scripts/slurm/launch.sh`

5 - Checkear que el source dentro de `scripts/slurm/work.sh` apunta al `cesga/vl3d_cesga_env.sh` del punto 1.

6 - Checkear que dentro de `scripts/slurm/launch.sh` se van a leer los .json del correspondiente experimento.

7 - Cambiad el email de `scripts/slurm/work.sh` por el vuestro.

8 - `./scripts/slurm/launch.sh`. Lanzar desde el directorio donde se encuentra el script.

9 - Que los dioses del Round-Robin estén con todos nosotros.


# Notas

## Experimento "Vegetation"
- Ruta modelo usado para vegetacion: /mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T3/pipe
- MODEL_ID='4'
- EXPERIMENT_NAME=VEGETATION