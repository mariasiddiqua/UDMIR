#!/bin/bash
#SBATCH --job-name=Navigation_Original_Model
#SBATCH --output=Navigation_Original_Model_%j.out 
#SBATCH --error=Navigation_Original_Model_%j.err   
#SBATCH --ntasks=1                           
#SBATCH --cpus-per-task=4                  
#SBATCH --mem=50G                            
#SBATCH --time=6-12:00:00                   
#SBATCH --account=rrg-menna
#SBATCH --gres=gpu:1                        

cd /lustre06/project/6080267/mariia/water/
module load opencv
source /lustre06/project/6080267/mariia/water/water/bin/activate
cd /lustre06/project/6080267/mariia/water/Original_Model

#srun python train.py -c config/nv_train.json
srun python train.py -c config/nv_test.json -p val