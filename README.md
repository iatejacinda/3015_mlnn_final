# Threatened Mammal Image Classification

A hierarchical deep learning image classification project for identifying selected threatened mammal taxa relevant to Singapore using real-world wildlife images from iNaturalist.

The project was developed in Jupyter Notebook using TensorFlow/Keras and follows the deep learning workflow from *Deep Learning with Python* by François Chollet.

## Overview

The aim of this project was to investigate how well a neural network restricted to fully connected `Dense` and `Dropout` layers could classify wildlife images.

Since convolutional neural networks were outside the scope of the project, images were resized, normalised and flattened before being passed into the neural networks.

Instead of using a single species classifier, the system follows a hierarchical taxonomic structure:

```text
Mammalia
├── Pholidota
│   └── Manidae
│       └── Sunda Pangolin
│
├── Rodentia
│   ├── Sciuridae
│   │   └── Red Giant Flying Squirrel
│   └── Hystricidae
│       └── East Asian/Malayan Porcupine
│
└── Carnivora
    ├── Felidae
    │   └── Mainland Leopard Cat
    │
    ├── Viverridae
    │   ├── Small-tooth Palm Civet
    │   ├── Masked Palm Civet
    │   └── Large Indian Civet
    │
    └── Mustelidae
        ├── Asian Small-clawed Otter
        └── Smooth-coated Otter
```

## Dataset

Images were collected using the iNaturalist API.

The final processed dataset contained **731 images**:

| Order | Images |
| --- | ---: |
| Pholidota | 99 |
| Rodentia | 140 |
| Carnivora | 492 |
| **Total** | **731** |

Where insufficient Singapore-only observations were available, images from the wider Southeast Asian region were included.

The dataset is imbalanced, so class weighting was used during training and **macro F1 score** was used as the main evaluation metric.

## Preprocessing

Each image was:

- converted to RGB
- resized and padded to `80 × 80`
- normalised to the range `0–1`
- flattened into `19,200` input features
- assigned order, family and species labels

A fixed NumPy snapshot of the processed dataset was used during modelling so that repeated experiments used the same data.

## Model Architecture

The final models use TensorFlow/Keras `Sequential` networks containing only `Dense` & `Dropout` layers.

Typical architecture:

```text
19,200 input features
        ↓
Dense(64, activation="tanh")
        ↓
Dropout(0.1)
        ↓
Softmax output
```

Training configuration:

- AdamW optimiser
- learning rate: `1e-4`
- sparse categorical cross-entropy
- batch size: `16`
- 50 epochs
- balanced class weights

## Hierarchical Classifiers

Five neural networks were trained:

1. **Order classifier**  
   Pholidota / Rodentia / Carnivora

2. **Rodentia family classifier**  
   Sciuridae / Hystricidae

3. **Carnivora family classifier**  
   Felidae / Viverridae / Mustelidae

4. **Viverridae species classifier**  
   Three civet species

5. **Mustelidae species classifier**  
   Two otter species

Branches containing only one possible child taxon were handled directly without training an unnecessary classifier.

## Experiments

The project investigated several factors affecting model performance:

- class imbalance
- class weighting
- dropout regularisation
- learning rate
- ReLU vs tanh activation
- prediction distributions
- confusion matrices
- class-level precision, recall & F1
- learning curves
- hidden-unit activation behaviour

Earlier ReLU experiments showed unstable behaviour, including majority-class collapse and severe hidden-unit inactivity.

Reducing the learning rate improved optimisation, while the final tanh configuration produced more balanced predictions across minority classes.

## Final Results

| Classifier | Validation Macro F1 | Test Macro F1 | Test Accuracy |
| --- | ---: | ---: | ---: |
| Order | 0.359 | 0.412 | 0.49 |
| Rodentia Family | 0.606 | 0.606 | 0.71 |
| Carnivora Family | 0.497 | 0.496 | 0.54 |
| Viverridae Species | 0.493 | 0.601 | 0.61 |
| Mustelidae Species | 0.701 | 0.618 | 0.66 |

The **Mustelidae species classifier** achieved the highest test macro F1 at approximately **0.62**.

The **Rodentia family classifier** achieved the highest test accuracy at approximately **71%**.

The Order classifier remained the weakest stage of the hierarchy, highlighting the difficulty of learning broad visual distinctions from flattened image pixels.

## Key Findings

The project showed that Dense-only neural networks can learn meaningful distinctions between some wildlife taxa, but performance varies considerably between branches.

Main limitations include:

- substantial class imbalance
- relatively small datasets for some branches
- loss of spatial information when images are flattened
- overfitting in some classifiers
- variation in lighting, viewpoint and background in real-world images
- propagation of errors through the hierarchical system

The model should therefore be treated as a **concept for assisted wildlife image classification**, rather than a production-ready conservation system.

## Conservation Context

A possible use of the system would be preliminary wildlife-image screening or pre-labelling.

Human verification would still be required, especially for rare taxa or uncertain predictions.

Because the system is hierarchical, an incorrect prediction at the Order or Family level prevents the image from reaching the correct downstream species classifier.

## Technologies Used

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- scikit-learn
- Pillow
- Matplotlib
- Requests
- iNaturalist API

## Repository Structure

```text
.
├── threatened_mammal_classification.ipynb
├── threatened_mammal_classification.html
├── README.md
├── .gitignore
└── data/
    └── .gitkeep
```

The processed NumPy dataset files are not included in the repository.

## Data

The fixed `.npy` files used for the reported experiments are kept locally and excluded from Git.

The notebook contains the original iNaturalist API acquisition & preprocessing pipeline.

To generate a new dataset locally, change:

```python
REBUILD_DATASET = False
```

to:

```python
REBUILD_DATASET = True
```

Because iNaturalist is a live data source, a newly generated dataset may differ from the fixed snapshot used for the reported results.

## Reproducibility

Model randomness was controlled using:

```python
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
```

The final experiments load a fixed saved dataset rather than rebuilding it from the API.

## Running the Project

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install the required Python packages and open the notebook using Jupyter:

```bash
jupyter notebook
```

To reproduce the exact reported experiments, the original processed `.npy` dataset snapshot is required.

Alternatively, a new dataset can be generated using the acquisition pipeline by setting:

```python
REBUILD_DATASET = True
```

## Suggested `.gitignore`

```gitignore
# Processed dataset
data/*.npy

# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc

# OS files
.DS_Store
Thumbs.db
```

## References

- Chollet, F. (2018). *Deep Learning with Python*. Manning Publications.
- iNaturalist. (2026). *iNaturalist API v2 Documentation*.
- TensorFlow Developers. (2026). *TensorFlow/Keras Documentation*.
- NumPy Developers. (2026). *NumPy Documentation*.
- scikit-learn Developers. (2026). *scikit-learn Documentation*.
- Pillow Contributors. (2026). *Pillow Documentation*.
- Matplotlib Development Team. (2026). *Matplotlib Documentation*.
- Requests Developers. (2026). *Requests Documentation*.

## About

This repository was created as part of a deep learning coursework project exploring hierarchical wildlife image classification under a Dense-network-only architectural constraint.
