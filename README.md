# Coffee Leaf Disease Detection

---

## Table of Contents
1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [Application Demo](#application-demo)
4.  [Motivation & Problem Statement](#motivation--problem-statement)
5.  [System Architecture](#system-architecture)
6.  [Technologies Used](#technologies-used)
7.  [Dataset](#dataset)
8.  [Core Functionalities](#core-functionalities)
    * [Standard Classification](#standard-classification)
    * [Few-Shot Learning](#few-shot-learning)
    * [Object Detection](#object-detection)
    * [Data Management](#data-management)
9.  [Setup and Installation](#setup-and-installation)
10. [Running the Application](#running-the-application)
11. [Directory Structure](#directory-structure)
12. [Usage](#usage)
13. [Results Highlights](#results-highlights)
---

## Overview

This project, "Coffee Leaf Disease Detection," is a web application designed to help identify coffee leaf diseases. The system uses deep learning for common diseases, incorporates few-shot learning techniques for rare or newly emerging diseases, and includes object detection to localize affected areas on coffee leaves.

The application is built with Streamlit, providing an interactive interface for image uploads, predictions, and managing disease class data.

---

## Key Features

* **Standard Disease Classification:** Accurately classifies common coffee leaf diseases (e.g., Rust, Phoma, Miner, Cercospora, Healthy) using a pre-trained EfficientNet-B0 model.
* **Few-Shot Learning for Rare Diseases:** Enables users to add new "rare" disease classes with only a few image samples (as few as 2-5). The system adapts using Prototypical Networks with a trainable projection layer, without needing to retrain the entire base model.
* **Object Detection:** Integrates a YOLO model (`best.pt`) to detect and draw bounding boxes around diseased regions on uploaded leaf images.
* **Dynamic Data Management:** Users can add new rare disease classes and upload corresponding images directly through the web interface. These are stored and used for few-shot learning.
* **Save/Load Few-Shot States:** Allows users to save the state of a trained few-shot model (including the feature extractor's projection layer and class prototypes) and load it later for continued use or further training.
* **Interactive Web Application:** A user-friendly interface built with Streamlit for easy interaction.

---

## Application Demo

Below is a glimpse of the application interface.

*(Replace this placeholder with an actual screenshot or GIF of your application. You can upload your image to GitHub or an image hosting service and link it here.)*

![Application Demo Placeholder](https://placehold.co/800x450/20232A/E0E0E0?text=App+Screenshot+Here&font=roboto)

*(Suggested caption: A brief look at the user interface for uploading images and viewing predictions.)*

---

## Motivation & Problem Statement

Coffee is a significant agricultural crop, and leaf diseases can severely impact its yield and quality. Traditional methods for disease diagnosis can be slow and subjective. Detecting rare or new diseases is particularly challenging due to the lack of extensive data for training conventional AI models.

This project aims to:
* Provide an accessible and efficient tool for diagnosing common coffee leaf diseases.
* Address the challenge of identifying rare diseases with limited data through few-shot learning.
* Offer localization of disease symptoms via object detection.
* Assist users in making informed decisions for crop health management.

---

## System Architecture

The system is designed as a modular web application:

* **User Interface (Streamlit):** Handles user interaction, image uploads, display of results, and management of rare classes and few-shot models.
* **Backend Logic (Python):**
    * **Standard Classifier:** Uses a pre-trained EfficientNet-B0.
    * **Feature Extractor for Few-Shot:** Uses the EfficientNet-B0 base with a trainable projection layer.
    * **Few-Shot Learning Module:** Implements Prototypical Networks.
    * **Object Detection Module:** Utilizes a YOLO model.
* **Data Storage:**
    * **Base Dataset:** For common diseases.
    * **Rare Dataset:** For user-added rare disease images.
    * **Saved Models:** For storing trained few-shot model states.

*(A general system architecture typically involves a frontend for user interaction, a backend for processing and model inference, and data storage for datasets and model files.)*

---

## Technologies Used

* **Programming Language:** Python 3.x
* **Core Libraries:**
    * `streamlit`: For the web application interface.
    * `torch` & `torchvision`: For deep learning model building, training, and inference.
    * `Pillow (PIL)`: For image processing.
    * `numpy`: For numerical operations.
    * `opencv-python (cv2)`: For image processing tasks, especially in object detection.
    * `ultralytics`: For YOLO object detection models.
    * `pandas`: For data handling (e.g., detection results).
    * `matplotlib`: For plotting.
    * `json`: For saving/loading metadata for few-shot models.
* **Deep Learning Models:**
    * **Classification:** EfficientNet-B0
    * **Few-Shot Learning:** Prototypical Networks with an EfficientNet-B0 backbone and a trainable projection layer.
    * **Object Detection:** YOLO (e.g., `model/best.pt`)
* **Development Environment:** Standard Python environment (virtual environment recommended).

---

## Dataset

The application uses two main categories of datasets:

1.  **Base Dataset (`App data/data_optimize/Basee_data/`):**
    * Contains images of common coffee leaf diseases and healthy leaves.
    * Classes typically include: `Cercospora`, `Healthy`, `Leaf rust`, `Miner`, `Phoma`.
    * This dataset is used to train the standard classifier and as the base knowledge for the few-shot learning feature extractor.

2.  **Rare Dataset (`App data/data_optimize/Rare data/`):**
    * This directory is for users to add images of new or rare disease classes.
    * Each subdirectory within `Rare data/` represents a new class.
    * Images here are used to train/adapt the few-shot learning model. The system requires a minimum number of images per new class (e.g., 4-5, based on `n_shot + n_query` parameters in the training script).

**Data Augmentation:** Various data augmentation techniques (brightness, contrast, flips, rotation, etc.) can be applied to improve model robustness.

---

## Core Functionalities

### Standard Classification
* Uses a pre-trained EfficientNet-B0 model (`model/efficientnet_coffee (1).pth`).
* Users can upload an image for disease class prediction with a confidence score.

### Few-Shot Learning
* **Strategy:** Prototypical Networks with a frozen EfficientNet-B0 backbone and a trainable linear projection layer.
* **Adding New Classes:** Users can define new rare disease classes and upload a small number of sample images via the "Add/Manage Rare Classes" interface.
* **Training:** The "Train Few-Shot Model" option initiates episodic training. Only the projection layer is trained.
* **Prediction:** If a few-shot model is active, predictions are made by comparing image embeddings (from the backbone + trained projection layer) to class prototypes.
* **Saving/Loading States:** Trained few-shot model states can be saved and reloaded.

### Object Detection
* Uses a YOLO model (`model/best.pt`) for detecting diseased regions.
* Outputs the image with bounding boxes, class labels, and confidence scores. A summary table is also displayed.

### Data Management
* **Adding/Deleting Rare Classes:** Users can manage rare class image data through the UI.
* **Managing Saved Few-Shot Models:** Users can load or delete saved few-shot model states.

---

## Setup and Installation

1.  **Prerequisites:**
    * Python 3.8+
    * pip (Python package installer)
    * Git (for cloning the repository)

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/coffee-leaf-disease-detection.git](https://github.com/your-username/coffee-leaf-disease-detection.git)
    cd coffee-leaf-disease-detection
    ```
    *(Replace `your-username` with your actual GitHub username if you fork/own it)*

3.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content (ensure versions are compatible with your Python environment and the script's needs):
    ```txt
    streamlit
    torch
    torchvision
    Pillow
    numpy
    opencv-python
    ultralytics
    pandas
    matplotlib
    # Add specific versions if known, e.g., streamlit==1.20.0
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download Model Weights (if not included in the repo):**
    * Ensure the standard classifier weights `model/efficientnet_coffee (1).pth` are present.
    * Ensure the YOLO detection model weights `model/best.pt` are present.
    * If these are large files, they might be hosted elsewhere (e.g., Git LFS, Google Drive). You'll need to download them into the `model/` directory.

6.  **Prepare Dataset Directories:**
    * Create the base dataset directory: `App data/data_optimize/Basee_data/`
        * Populate it with subdirectories for each common disease class (e.g., `Healthy`, `Rust`, `Miner`, `Cercospora`, `Phoma`), each containing respective images.
    * Create the rare dataset directory: `App data/data_optimize/Rare data/` (This can be initially empty).
    * Create the saved models directory: `saved_few_shot_models/` (This will be initially empty).

---

## Running the Application

Once the setup is complete, run the Streamlit application from the project's root directory:

```bash
streamlit run your_script_name.py
```
*(Replace `your_script_name.py` with the actual name of your main Python script, e.g., `app.py` or `main.py`)*

The application should open in your default web browser.

---

## Directory Structure

A typical directory structure might look like this:

```
coffee-leaf-disease-detection/
├── App data/
│   └── data_optimize/
│       ├── Basee_data/       # Base classes dataset
│       │   ├── Healthy/
│       │   ├── Rust/
│       │   └── ...
│       └── Rare data/          # Rare classes added by user
│           ├── New_Disease_1/
│           └── ...
├── model/
│   ├── efficientnet_coffee (1).pth # Standard classifier weights
│   └── best.pt                     # YOLO detection model weights
├── saved_few_shot_models/        # Saved few-shot model states
│   └── MySavedState1/
│       ├── feature_extractor_state_dict.pth
│       ├── prototypes.pt
│       └── metadata.json
├── your_script_name.py           # Main Streamlit application script
├── assets/                       # Optional: For storing images like your demo screenshot
│   └── app_demo.png
├── requirements.txt
└── README.md
```
*(Added an optional `assets/` directory suggestion for your demo image)*

---

## Usage

1.  **Launch the application:** Follow the "Running the Application" steps.
2.  **Choose an action from the main panel:**
    * **Upload & Predict:** Upload an image to get a classification (either standard or few-shot, depending on the active mode).
    * **Add/Manage Rare Classes:**
        * Add new disease classes by providing a name and uploading sample images.
        * Delete existing rare classes.
    * **Train Few-Shot Model:**
        * Initiate the training process for the few-shot learning module using the base dataset and any added rare classes.
        * After training, you can save the model state.
    * **Detection:** Upload an image to perform object detection and see bounding boxes on diseased areas.
3.  **Sidebar Options:**
    * **Reset to Standard Classifier:** Reverts to using the standard classifier and clears any active few-shot model state.
    * **Load/Delete Saved Few-Shot Models:** Manage your saved few-shot learning states.

---

## Results Highlights

* The standard classifier (EfficientNet-B0) is expected to achieve high accuracy (e.g., >99%) on common diseases, especially after data augmentation.
* The few-shot learning module, particularly with the projection layer, demonstrates the ability to learn new classes from very few samples (2, 5, or 10 shots) with good performance.
* YOLO models provide effective localization of diseased areas.

*(Refer to the "Results and Discussion" section of your project report for specific metrics and detailed analysis.)*

---



