#!/bin/bash -l
#SBATCH --cluster=genius                        
#SBATCH --partition=gpu_p100                
#SBATCH --gpus-per-node=1                       
#SBATCH --nodes=1                               
#SBATCH --time=00:30:00
#SBATCH --account=lp_edu_vives_ti_deep_learning
#SBATCH --ntasks=9

set -euo pipefail

# Zorg dat je venv-functie beschikbaar is
source ~/.bashrc
activate_tensorflow

# Ga naar de folder met train/ val/ test/ en je python script
cd $VSC_DATA/image_classification/alle_schilders

# Extra debug info in de output
echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
python -c "import sklearn; import pandas; print('sklearn/pandas OK')"

# Run jouw training script (schrijft automatisch naar out_simple_convnet/)
python vgg16_finetune_class_weights.py

echo "DONE. Results are in:"
echo "$(pwd)/out_vgg16_finetune_class_weights"
