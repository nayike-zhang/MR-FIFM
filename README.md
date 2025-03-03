# MR-FIFM
The code for the paper **Multimodal Hand Feature Recognition Base on Modality-Related Feature Interaction**

## Introduction
This project implements a **multi-modal hand biometric fusion recognition system** that combines multiple hand-based biometric features (e.g., fingerprint, palmprint, and vein patterns) to improve recognition accuracy.

#### Requirements
pip install -r requirements.txt

#### Training a Model
To train the model, run the following command:
`python train.py`

You can customize the training process using the following arguments in 'veri_mm.yml':

--IMS_PER_BATCH: Batch size for training.

--MAX_EPOCHS: Number of training epochs.

--BASE_LR: Learning rate.
