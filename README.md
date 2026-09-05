# MNIST Digit Classification: Custom CNN vs. LeNet-5 Transfer Learning

A comparative study of two deep learning approaches to handwritten digit classification on the MNIST dataset — a custom Convolutional Neural Network trained from scratch, and a LeNet-5 architecture trained via a pre-train → freeze → fine-tune transfer learning workflow.

## Overview

Both models are trained and evaluated on the same MNIST train/validation/test splits to allow a fair, apples-to-apples comparison. The notebook covers the full pipeline: data loading from raw IDX binary files, preprocessing, class imbalance analysis, data augmentation, model training, and evaluation.

**Key techniques demonstrated:**
- Custom parsing of the raw MNIST IDX binary format (no pre-built dataset loaders)
- Class distribution analysis and balanced class-weighting to handle imbalance
- Data augmentation (rotation, shift, zoom) via `ImageDataGenerator`
- Early stopping with best-weight restoration
- Transfer learning workflow: pre-train LeNet-5 from scratch → save checkpoint → reload → freeze base layers → fine-tune classifier head at a low learning rate
- Evaluation via test accuracy, precision/recall/F1, confusion matrices, and misclassified-sample inspection

## Results

| Model | Test Loss | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Custom CNN** | 0.0164 | **99.48%** | 0.9948 | 0.9948 | 0.9948 |
| LeNet-5 (Transfer Learning, fine-tuned) | 0.0872 | 97.34% | 0.9734 | 0.9734 | 0.9734 |

**Takeaway:** The custom CNN trained from scratch outperformed the LeNet-5 transfer learning approach on this task. This is expected for MNIST — a relatively simple, well-behaved dataset — where a deeper, purpose-built architecture (32 → 64 filter CNN with dropout) has enough capacity and data to learn strong features directly, while freezing LeNet-5's base layers during fine-tuning limits how much the smaller, shallower architecture can adapt.

## Project Structure

- `mnist_cnn_vs_lenet_transfer_learning.ipynb` — full notebook: data pipeline, both models, training, evaluation, and comparison

## Tech Stack

Python · TensorFlow / Keras · NumPy · Pandas · scikit-learn · Matplotlib · Seaborn

## How to Run

1. Download the MNIST dataset in IDX format (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`), e.g. from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/).
2. Update the file paths at the top of the notebook to point to your local copies.
3. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
   ```
4. Run all cells in order (`Kernel > Restart & Run All`).

