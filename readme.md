# Dynamically Expandable Interactive Trajectory Predictor (DEITP)

## Introduction
This is the official implementation of the paper [*Toward Zero-forgetting Continual Learning for Interactive Trajectory Prediction: A Dynamically Expandable Approach*]().

## Dependencies
The conda environment can be created by the following command

```
conda create -n DEITP python=3.8
conda activate DEITP
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset
Dataset used in this work comes from publicly available [*INTERACTION*](https://interaction-dataset.com/) datasets, and the preprocessing method follows the [*D-GSM*](https://github.com/BIT-Jack/D-GSM). The original dataset used in the experiment can be downloaded from [*here*](https://drive.google.com/drive/folders/14yPZF6P146HA0CNwhwESYT4-zBlALSr_?usp=drive_link). Please download the dataset and put it under the `data/original` folder. Training scripts will automatically process the data and save processed data in the `data/processed` folder.

## Usage

### Training
The repository provides training scripts for several continual learning approaches in continuous traffic scenarios. DEITP, called as Dynamically Expandable Model (DEM) for simplicity in the code, will be trained by running the following command:
```
bash scripts/train_dem.sh
```
We also propose a familiarity autoencdoer (FAE) based approach for task detection in the task-free continual learning (TFCL) setting. The FAE-based approach can be trained by running the following command:
```
bash scripts/train_fae.sh
```




### Testing

## Acknowledgement
We sincerely appreciate the following github repos for their valuable code base we build upon:
- [https://github.com/BIT-Jack/D-GSM](https://github.com/BIT-Jack/D-GSM)