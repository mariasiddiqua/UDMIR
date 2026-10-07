#!/bin/bash
#SBATCH --job-name=Restore
#SBATCH --output=Restore_%j.out 
#SBATCH --error=Restore_%j.err   
#SBATCH --ntasks=1                           
#SBATCH --cpus-per-task=4                  
#SBATCH --mem=50G                            
#SBATCH --time=6-12:00:00                   
#SBATCH --account=def-menna
#SBATCH --gres=gpu:1                        

cd /maria/udmir/
module load opencv
source /maria/udmir/bin/activate
cd maria/udmir
srun python train.py -c config/uw_train.json
#srun python train.py -c config/uw_test.json