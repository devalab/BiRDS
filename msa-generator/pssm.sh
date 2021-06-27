#!/bin/bash
#SBATCH --account=hiv
#SBATCH --qos=normal
# #SBATCH --account=research
# #SBATCH --qos=medium
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2000
#SBATCH --mincpus=24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
##SBATCH --nodelist=node15
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END

rm -rf /tmp/$USER
USER_DIR=/scratch/$USER
# rm -rf $USER_DIR/pssm_generation/data/scPDB
mkdir -p $USER_DIR
cd $USER_DIR
ls $USER_DIR | grep -v pssm_generation | xargs rm -rf
ADA_FOLDER=/share2/$USER/data/pssm_generation

rsync -av --delete --exclude='data/scPDB/' ada:$ADA_FOLDER $USER_DIR/
cd pssm_generation
# export INSTALL_DIR=$USER_DIR/pssm_generation/hhsuite2
# export HHLIB=$INSTALL_DIR/lib/hh
# export PATH=$PATH:$INSTALL_DIR/bin:$HHLIB/scripts
# export LD_LIBRARY_PATH=$INSTALL_DIR/lib
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$USER_DIR/pssm_generation/hhsuite2/bin/boost/lib

rsync -av --include="*/" --include-from=$USER_DIR/pssm_generation/splits/$1 --exclude="*" ada:$ADA_FOLDER/data/scPDB $USER_DIR/pssm_generation/data
srun -N 1 -n 1 python calculate_pssm.py $1 -ncpu=40 &
wait
rsync -av --include="*/" --include-from=$USER_DIR/pssm_generation/splits/$1 --exclude="*" $USER_DIR/pssm_generation/data/scPDB ada:$ADA_FOLDER/data
