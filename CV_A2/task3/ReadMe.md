# Assignment 2 Task 3

## Project Structure

The project integrates the provided original code with custom scripts for pipeline automation and foundation model evaluation.

### Original Provided Files
* `depth_model.py`: Defines the ResNet50-based monocular depth estimation model.
* `scannet_dataset.py`: Dataloader for the ScanNet dataset handling RGB and depth maps.
* `metrics.py`: Computes Absolute Relative Error (AbsRel) and performs scale-shift alignment.
* `train.py`: Training script optimizing the model using SiLog loss.
* `test.py`: Evaluation script for testing trained checkpoints on the validation split.

### Newly Added Files
* `data/make_dummy_scannet.py`: Generates dummy images and 16-bit PNG depth maps to test the data pipeline locally.
* `test_foundation.py`: Implements the evaluation pipeline for pre-trained 3D foundation models (Depth Anything 3 and VGGT), handling ViT-specific resolution adjustments (multiples of 14) and tensor reshaping.
* `scripts/run.sh`: Automates the entire Task 3 pipeline in 3 steps: (1) Trains and evaluates the baseline model; (2) Executes the data scale ablation study across 4 configurations; (3) Evaluates pre-trained DA3 and VGGT foundation models.
* `plot_results.py`: Draw the plot of task3.
* `scripts/generate_plots.sh`: Shell script to automate the execution of the plotting process.


## Execution

### Step 1: Data preparation
Ensure that the local `scannet.tar.gz` file has been uploaded to the directory `~/CV_A2/task3`.

Then extract the archive with the following command:  

```bash
tar -xzf scannet.tar.gz -C ./data/
```

### Step 2: Run the Pipeline
Execute the master script to run the baseline training, ablation studies, and foundation model evaluations sequentially.

```bash
nohup bash scripts/run.sh > results/all_execution.log 2>&1 &
```
Monitor the overall progress:
```bash
tail -f results/all_execution.log
```
#### Quantitative Results:
All output JSON metrics will be automatically saved in `results/baseline/` and `results/ablation/`.

### Step 3: Plot
Execute the plotting script in the background.

```bash
nohup bash scripts/generate_plots.sh > results/plot.log 2>&1 &
```

Monitor the plotting progress:
```bash
tail -f results/plot.log
```

