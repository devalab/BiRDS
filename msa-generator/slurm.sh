#!/bin/bash
#SBATCH --account=hiv
#SBATCH --qos=normal
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2000
#SBATCH --mincpus=24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
# #SBATCH --nodelist=node15
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END

USER_DIR=/scratch/$USER
PROJECT_NAME=msa-generator
DATASET=scPDB
NCPUS=24

# rm -rf /tmp/$USER
# rm -rf $USER_DIR/$PROJECT_NAME/data/$DATASET
mkdir -p $USER_DIR/$PROJECT_NAME/data/$DATASET
# ls $USER_DIR | grep -v $PROJECT_NAME | xargs rm -rf
STORE_FOLDER=/share2/$USER/data/$PROJECT_NAME

rsync -av --exclude=data/$DATASET/ $STORE_FOLDER $USER_DIR/
cd $USER_DIR/$PROJECT_NAME
# export INSTALL_DIR=$USER_DIR/$PROJECT_NAME/hhsuite2
# export HHLIB=$INSTALL_DIR/lib/hh
# export PATH=$PATH:$INSTALL_DIR/bin:$HHLIB/scripts
# export LD_LIBRARY_PATH=$INSTALL_DIR/lib
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$USER_DIR/$PROJECT_NAME/hhsuite2/bin/boost/lib

rsync -av --include="*/" --include-from=$USER_DIR/$PROJECT_NAME/splits/$1 --exclude="*" $STORE_FOLDER/data/$DATASET $USER_DIR/$PROJECT_NAME/data
find $USER_DIR/$PROJECT_NAME/data/$DATASET -type d -empty -delete
srun -N 1 -n 1 python calculate_pssm.py $1 -ncpu=$NCPUS &
wait
rsync -av --include="*/" --include-from=$USER_DIR/$PROJECT_NAME/splits/$1 --exclude="*" $USER_DIR/$PROJECT_NAME/data/$DATASET $STORE_FOLDER/data
