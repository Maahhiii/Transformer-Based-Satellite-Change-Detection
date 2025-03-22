# Enhanced Satellite Imagery Change Detection using Transformer-Based Deep Learning Models

Satellite image analysis is crucial for monitoring land use changes, urbanization, and environmental management. This study evaluates four transformer-based deep learning models for satellite-based land use change detection: **Dual Encoder Transformer (DET) with U-Net**, **Spatial-Temporal Attention Transformer (STA) with U-Net**, **Vision Transformer (ViT) with U-Net**, and **Multi-Scale Vision Transformer (MSViT) with U-Net**.

## Models and Code

The repository contains the implementations of the following models:

1. **Dual Encoder Transformer (DET) with U-Net**  
   - Training Script: `src/Dual_Encoder_Transformer/train.py`  
   - Testing Script: `src/Dual_Encoder_Transformer/test.py`

2. **Spatial-Temporal Attention Transformer (STA) with U-Net**  
   - Training Script: `src/Spatial_Temporal_Attention/train.py`  
   - Testing Script: `src/Spatial_Temporal_Attention/test.py`

3. **Vision Transformer (ViT) with U-Net**  
   - Training Script: `src/Vision_Transformer/train.py`  
   - Testing Script: `src/Vision_Transformer/test.py`

4. **Multi-Scale Vision Transformer (MSViT) with U-Net**  
   - Training Script: `src/Multi_Scale_Vision_Transformer/train.py`  
   - Testing Script: `src/Multi_Scale_Vision_Transformer/test.py`

Each script is designed to train and test the respective model on satellite imagery datasets.

## Dependencies

To run the code, ensure you have the following dependencies installed:

- **Python 3.x**
- **TensorFlow**
- **NumPy**
- **Matplotlib**
- **OpenCV**
- **Scikit-learn**

A `requirements.txt` file is provided with the necessary packages. You can install them using:

pip install -r requirements.txt

## Dataset

This study uses a satellite image dataset stored in the following directory: `dataset/CLCD/`

## Citation

If you use this repository or its contents in your research, please cite the following paper:

**"Enhanced Satellite Imagery Change Detection using Transformer-Based Deep Learning Models"**  
Published in: **The Visual Computer**  
Authors: Kathan Patel, Mahi Kachhadiya, Rashmi Bhattad, Deepak Patel  
DOI: [not available]

## DOI for Repository

This repository is assigned a DOI: `https://doi.org/10.5281/zenodo.15066892`
