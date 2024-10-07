#!/bin/bash
#BSUB -n 360                    # Request 360 cores
#BSUB -R "span[ptile=72]"        # Specify 72 cores per node
#BSUB -q p_short                 # Use the p_short queue
#BSUB -W 1:00                    # Walltime of 1 hour
#BSUB -P R000                    # Project code
#BSUB -x                         # Exclusive mode
#BSUB -J land_emul                # Job name
#BSUB -o emul.out.%J       # Standard output file
#BSUB -e emul.err.%J       # Standard error file
#BSUB -app spreads_filter        # Application profile
#BSUB -I                         # Run interactively

echo "Running Random Forest training on Juno"

# Load required modules (example)
#conda activate pytorch_gpu_horovod_lightning

# Activate the Python environment (if needed)
#source /data/cmcc/$USER/d4o/install/INTEL/source.me

# Run the Python script
#same day only atmo forcing training
python /work/cmcc/lg07622/work/emulator/RF_SD_par.py

#same and previous day atmo forcing training
#python /work/cmcc/lg07622/work/emulator/RF_PD_par.py
