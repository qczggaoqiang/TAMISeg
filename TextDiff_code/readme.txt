TextDiff Training

Step 1: Save the trained weights from the `run_files` folder of the first training phase to the `checkpoints` folder in `TextDiff_code`. There is already an example file in this folder; simply replace it when the time comes (the filename must be the same).
Step 2: Training
python train.py --exp experiments/qata_cov19_v2_2/condseg.json
The training results will be saved in the `saved_textdiff` folder within the current directory.
