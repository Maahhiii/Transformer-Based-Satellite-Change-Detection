# Enhanced Satellite Imagery Change Detection using Transformer-Based Deep Learning Models

Satellite image analysis is crucial for monitoring land use changes, urbanization, and environmental management. This study evaluates four transformer-based deep learning models for satellite-based land use change detection: **Dual Encoder Transformer (DET) with U-Net**, **Spatial-Temporal Attention Transformer (STA) with U-Net**, **Vision Transformer (ViT) with U-Net**, and **Multi-Scale Vision Transformer (MSViT) with U-Net**.

## Models and Code

- **Dual Encoder Transformer (DET) with U-Net**: Implemented in `src/DET/DET_UNet_Train.py` and `src/DET/DET_UNet_Test.py`.
- **Spatial-Temporal Attention Transformer (STA) with U-Net**: Implemented in `src/STA/STA_UNet_Train.py` and `src/STA/STA_UNet_Test.py`.
- **Vision Transformer (ViT) with U-Net**: Implemented in `src/ViT/ViT_UNet_Train.py` and `src/ViT/ViT_UNet_Test.py`.
- **Multi-Scale Vision Transformer (MSViT) with U-Net**: Implemented in `src/MSViT/MSViT_UNet_Train.py` and `src/MSViT/MSViT_UNet_Test.py`.

## Dependencies

To run the code, ensure you have the following dependencies installed:

- **Python 3.x**
- **TensorFlow**
- **NumPy**
- **Matplotlib**
- **OpenCV**
- **Scikit-learn**

A `requirements.txt` file is provided with the necessary packages. You can install them using:

- **pip install -r requirements.txt**

## Setup and Usage

1. **Clone the Repository**: Use `git clone https://github.com/your-username/your-repo-name.git`.
2. **Install Dependencies**: Run `pip install -r requirements.txt`.
3. **Train Models**: Execute each model's training script (e.g., `python src/DET/DET_UNet_Train.py`).
4. **Test Models**: Use the testing scripts to evaluate model performance (e.g., `python src/DET/DET_UNet_Test.py`).

## Dataset

This study uses a satellite image dataset. For access to the dataset go to `dataset/CLCD`.

## Citation

If you use this repository or its contents in your research, please cite the following paper:

**"Enhanced Satellite Imagery Change Detection using Transformer-Based Deep Learning Models"**  
Published in: **The Visual Computer**  
Authors: Kathan Patel, Mahi Kachhadiya, Rashmi Bhattad, Deepak Patel  
DOI: -

## DOI for Repository

This repository is assigned a DOI: [Insert DOI if available].




