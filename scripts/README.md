# Ejecución clasificación PNOA2 en Galicia

1 - Cambiar las rutas de las variables `VL3D_DIR` y `VL3D_SCRIPT` en `cesga/vl3d_cesga_env.sh` para que apunten a este repositorio y hacer source del script. También VL3D_ENV para que apunte a vuestro entorno de conda para este proyecto.

2 - Definir rutas **absolutas** al modelo elegido para el experimento. Deben definirse en el script `scripts/experiment_generation/replace_paths.sh`. 

3 - Ejecutar el script pasando con -i el directorio (`experiments/nombre_experimento/nombre_usuario`) donde están todos los .json generados que se van a ejecutar.

3 - `chmod +x scripts/slurm/work.sh scripts/slurm/launch.sh`

4 - Checkear que el source dentro de `scripts/slurm/work.sh` apunta al `cesga/vl3d_cesga_env.sh` del punto 1.

5 - Checkear que dentro de `scripts/slurm/launch.sh` se van a leer los .json del correspondiente experimento.

6 - Cambiad el email de `scripts/slurm/work.sh` por el vuestro.

6 - `./scripts/slurm/launch.sh`

7 - Que los dioses del Round-Robin estén con todos nosotros.


# Notas

- Ruta modelo usado para vegetacion: /mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T3/pipe
