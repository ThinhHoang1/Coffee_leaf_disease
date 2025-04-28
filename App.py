# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image, ImageOps # Added ImageOps
import os
import tempfile
import random
import shutil
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import cv2
# from PIL import Image # Already imported
from ultralytics import YOLO
import pandas as pd
import json # Added for saving/loading metadata

st.set_page_config(layout="wide")

# --- Constants and Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVED_MODELS_DIR = "saved_few_shot_models"
BASE_DATASET = "App data/data_optimize/Basee_data"
RARE_DATASET = "App data/data_optimize/Rare data/"
MODEL_WEIGHTS_PATH = "model/efficientnet_coffee (1).pth" # For Standard Classifier and initial feature extractor state
YOLO_MODEL_PATH = "model/best.pt" # For Detection

# Ensure directories exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RARE_DATASET, exist_ok=True) # Create rare data dir if not present

# Check required files/folders exist
if not os.path.isdir(BASE_DATASET):
    st.error(f"Base dataset directory not found: {BASE_DATASET}")
    st.stop()
if not os.path.isfile(MODEL_WEIGHTS_PATH):
    st.error(f"Classifier weights file not found: {MODEL_WEIGHTS_PATH}")
    st.stop()
if not os.path.isfile(YOLO_MODEL_PATH):
    st.error(f"YOLO detection model file not found: {YOLO_MODEL_PATH}")
    st.stop()

st.sidebar.info(f"Using device: {DEVICE}")
TEMP_DIR = tempfile.mkdtemp() # For temporary files if needed


# --- Helper Functions for Saving/Loading Few-Shot States ---

def list_saved_models():
    """Returns a list of names of saved few-shot model states."""
    if not os.path.isdir(SAVED_MODELS_DIR):
        return []
    # List only directories within the SAVED_MODELS_DIR
    return [d for d in os.listdir(SAVED_MODELS_DIR) if os.path.isdir(os.path.join(SAVED_MODELS_DIR, d))]

def save_few_shot_state(name, model, prototypes, proto_labels, current_class_names):
    """Saves the model state, prototypes, and metadata."""
    if not name or not name.strip():
        st.error("Please provide a valid name for the saved model.")
        return False
    # Sanitize name for directory creation
    sanitized_name = "".join(c for c in name if c.isalnum() or c in ('_', '-')).rstrip()
    if not sanitized_name:
        st.error("Invalid name after sanitization. Use letters, numbers, underscore, or hyphen.")
        return False

    save_dir = os.path.join(SAVED_MODELS_DIR, sanitized_name)

    # Handle existing directory (Ask for overwrite confirmation)
    if os.path.exists(save_dir):
         # Use columns for better layout of overwrite confirmation
        col1, col2 = st.columns([3,1])
        with col1:
            st.warning(f"Model name '{sanitized_name}' already exists.")
        with col2:
            # Use a unique key for the overwrite button based on the name
            overwrite_key = f"overwrite_{sanitized_name}"
            if st.button("Overwrite?", key=overwrite_key):
                 st.info(f"Overwriting '{sanitized_name}'...")
                 # Proceed with saving below
            else:
                 st.info("Save cancelled. Choose a different name or click 'Overwrite?'.")
                 return False # Stop if not confirmed overwrite
    else:
        st.info(f"Saving new model state '{sanitized_name}'...")


    try:
        os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist or was confirmed for overwrite

        # 1. Save Model State Dictionary (ensure model is on CPU before saving state_dict for better compatibility)
        model.to('cpu') # Move model to CPU
        model_path = os.path.join(save_dir, "feature_extractor_state_dict.pth")
        torch.save(model.state_dict(), model_path)
        model.to(DEVICE) # Move model back to original device

        # 2. Save Prototypes Tensor (move to CPU before saving)
        prototypes_path = os.path.join(save_dir, "prototypes.pt")
        torch.save(prototypes.cpu(), prototypes_path)

        # 3. Save Metadata (Labels and Class Names active during training)
        metadata = {
            "prototype_labels": proto_labels, # Should be a standard list
            "class_names_on_save": current_class_names # List of strings
        }
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        st.success(f"Few-shot model state '{sanitized_name}' saved successfully!")
        return True
    except Exception as e:
        st.error(f"Error saving model state '{sanitized_name}': {e}")
        st.exception(e) # Show full traceback for debugging
        # Clean up potentially partially saved directory if error occurred AFTER creation/overwrite confirmation
        if os.path.exists(save_dir):
             try:
                 shutil.rmtree(save_dir)
                 st.info(f"Cleaned up partially saved directory '{save_dir}'.")
             except Exception as cleanup_e:
                 st.error(f"Error cleaning up directory during save failure: {cleanup_e}")
        return False

def load_few_shot_state(name, model_to_load_into, current_class_names):
    """Loads a saved model state, prototypes, and labels into session state and the model."""
    load_dir = os.path.join(SAVED_MODELS_DIR, name)
    if not os.path.isdir(load_dir):
        st.error(f"Saved model directory '{load_dir}' not found.")
        return False

    model_path = os.path.join(load_dir, "feature_extractor_state_dict.pth")
    prototypes_path = os.path.join(load_dir, "prototypes.pt")
    metadata_path = os.path.join(load_dir, "metadata.json")

    if not all(os.path.exists(p) for p in [model_path, prototypes_path, metadata_path]):
        st.error(f"Saved model '{name}' is incomplete. Files missing in '{load_dir}'.")
        return False

    try:
        # 1. Load Metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        loaded_proto_labels = metadata.get("prototype_labels")
        saved_class_names = metadata.get("class_names_on_save")

        if loaded_proto_labels is None or saved_class_names is None:
             st.error(f"Metadata file for '{name}' is corrupted or missing required keys.")
             return False

        # **CRUCIAL CHECK**: Compare saved class names with current class names
        if set(saved_class_names) != set(current_class_names):
            st.warning(f"⚠️ **Class Mismatch!**")
            st.warning(f"Saved model '{name}' classes: `{saved_class_names}`")
            st.warning(f"Current active classes: `{current_class_names}`")
            st.warning("Predictions might be incorrect or errors may occur. Proceed with caution.")
            # Decide whether to proceed or stop. Let's proceed with warning.

        # 2. Load Model State Dictionary
        # Ensure model architecture is correct *before* loading state dict
        model_to_load_into.to(DEVICE) # Move model to target device
        state_dict = torch.load(model_path, map_location=DEVICE) # Load state dict to target device
        try:
             # Use strict=False initially if unsure about architecture changes
             missing_keys, unexpected_keys = model_to_load_into.load_state_dict(state_dict, strict=True)
             if missing_keys: st.warning(f"Loaded state dict is missing keys: {missing_keys}")
             if unexpected_keys: st.warning(f"Loaded state dict has unexpected keys: {unexpected_keys}")
        except RuntimeError as e:
             st.error(f"RuntimeError loading state_dict for '{name}'. Architecture mismatch?")
             st.error(e)
             return False
        model_to_load_into.eval() # Set to evaluation mode after loading

        # 3. Load Prototypes (load directly to target device)
        loaded_prototypes = torch.load(prototypes_path, map_location=DEVICE)

        # 4. Update Session State
        st.session_state.final_prototypes = loaded_prototypes
        st.session_state.prototype_labels = loaded_proto_labels
        st.session_state.few_shot_trained = True # Mark as trained since we loaded a state
        st.session_state.model_mode = 'few_shot' # Switch to few-shot mode

        st.success(f"Successfully loaded few-shot model state '{name}'. Mode set to Few-Shot.")
        return True

    except Exception as e:
        st.error(f"Error loading model state '{name}': {e}")
        st.exception(e)
        # Optional: Reset state if loading fails partially
        # st.session_state.final_prototypes = None
        # st.session_state.prototype_labels = None
        # st.session_state.few_shot_trained = False
        # st.session_state.model_mode = 'standard'
        return False

def delete_saved_model(name):
    """Deletes a saved model directory."""
    delete_dir = os.path.join(SAVED_MODELS_DIR, name)
    if not os.path.isdir(delete_dir):
        st.error(f"Cannot delete. Saved model '{name}' not found.")
        return False
    try:
        shutil.rmtree(delete_dir)
        st.success(f"Deleted saved model '{name}'.")
        return True
    except Exception as e:
        st.error(f"Error deleting saved model '{name}': {e}")
        return False


# --- Model Architectures ---

# EfficientNet feature extractor with projection layer
class EfficientNetWithProjection(nn.Module):
    def __init__(self, base_model, output_dim=1024):
        super(EfficientNetWithProjection, self).__init__()
        self.model = base_model # This holds the EfficientNet base (feature extractor part)
        # Determine the input feature size dynamically from the base model if possible
        # For EfficientNetB0, the layer before the classifier has 1280 features
        in_features = 1280 # Hardcoding for effnetb0 is reliable here
        self.projection = nn.Linear(in_features, output_dim) # Projection layer

    def forward(self, x):
        features = self.model(x) # Get features from EfficientNet base
        return self.projection(features) # Project to output_dim dimensions

# Base EfficientNet model structure (used for loading weights)
def get_base_efficientnet_architecture(num_classes=5):
    # Load architecture only, ensure it matches the saved weights structure
    model = models.efficientnet_b0(weights=None) # Start with no pretrained weights here
    in_features = model.classifier[1].in_features # Get feature dimension
    model.classifier[1] = nn.Linear(in_features, num_classes) # Adjust final layer to match saved model
    return model

# Feature Extractor model structure (for few-shot)
def get_feature_extractor_base():
    # This function prepares the base model, loads coffee weights, then removes classifier
    # Start with the architecture matching the saved weights (5 classes)
    base_model = get_base_efficientnet_architecture(num_classes=5)

    # Load the coffee-specific weights into this matching architecture
    try:
        # Use weights_only=True for security unless the model itself is saved
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE) #, weights_only=True) # Set weights_only based on how model was saved
        # Load weights strictly as the architecture matches exactly
        missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=True)
        # No warnings needed if strict=True and it passes
    except Exception as e:
        st.error(f"Error loading model weights from {MODEL_WEIGHTS_PATH} into base architecture: {e}")
        st.exception(e)
        st.stop()

    # Remove the classifier to use it as a feature extractor
    base_model.classifier = nn.Identity()
    base_model.eval() # Set base model to eval mode
    return base_model

# Function to load the standard classifier model (for reset/fallback)
def load_standard_classifier():
    # This loads the model intended for standard 5-class classification
    model = get_base_efficientnet_architecture(num_classes=5) # Get the 5-class architecture
    try:
        # Use weights_only=True for security if appropriate
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE) #, weights_only=True)
        model.load_state_dict(state_dict, strict=True) # Strict loading
    except Exception as e:
        st.error(f"Error loading model weights for standard classifier: {e}")
        st.exception(e)
        st.stop()
    model.to(DEVICE)
    model.eval()
    return model


# --- Caching ---
# Cache the feature extractor model resource (Base + Projection)
@st.cache_resource
def cached_feature_extractor_model():
    # This function creates the base, loads weights, removes classifier, then adds projection
    base_model = get_feature_extractor_base()
    model = EfficientNetWithProjection(base_model, output_dim=1024)
    model.to(DEVICE)
    model.eval()
    st.sidebar.info("Feature extractor model ready (cached).")
    return model

# Cache the standard classifier model resource
@st.cache_resource
def cached_standard_classifier():
    model = load_standard_classifier()
    st.sidebar.info("Standard classifier model ready (cached).")
    return model

# Cache data loading and processing
@st.cache_data
def get_combined_dataset_and_indices(base_path, rare_path):
    try:
        # Define transform inside function or ensure it's globally defined before call
        transform_local = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load base dataset
        full_dataset = datasets.ImageFolder(base_path, transform_local)
        num_base_classes = len(full_dataset.classes)
        base_class_names = sorted(full_dataset.classes) # Get base names sorted

        # Load rare dataset if it exists and has content
        rare_classes_found = 0
        rare_class_names = []
        if os.path.isdir(rare_path) and any(os.scandir(rare_path)): # Check if dir exists and is not empty
             try:
                 rare_dataset = datasets.ImageFolder(rare_path, transform_local)
                 if len(rare_dataset.samples) > 0:
                     # IMPORTANT: Adjust labels for rare classes to start after base classes
                     rare_dataset.samples = [(path, label + num_base_classes) for path, label in rare_dataset.samples]
                     combined_dataset = ConcatDataset([full_dataset, rare_dataset])
                     rare_classes_found = len(rare_dataset.classes)
                     rare_class_names = sorted(rare_dataset.classes) # Get rare names sorted
                 else:
                     combined_dataset = full_dataset # Rare dir exists but no samples
             except Exception as e_rare:
                 st.warning(f"Could not load rare dataset from {rare_path}: {e_rare}. Using base dataset only.")
                 combined_dataset = full_dataset
        else:
            combined_dataset = full_dataset # Rare dir doesn't exist or is empty

        # Create class indices mapping (label -> list of dataset indices)
        indices = {}
        current_idx = 0
        # Iterate through the combined dataset structure correctly
        if isinstance(combined_dataset, ConcatDataset):
            for ds in combined_dataset.datasets:
                for _, label in ds.samples: # We only need the label here
                    indices.setdefault(label, []).append(current_idx)
                    current_idx += 1
        else: # Only base dataset
             for idx, (_, label) in enumerate(combined_dataset.samples):
                 indices.setdefault(label, []).append(idx)


        # Combine class names in the correct order
        class_names = base_class_names + rare_class_names

        # Display stats in sidebar
        st.sidebar.metric("Base Classes", num_base_classes)
        st.sidebar.metric("Rare Classes", rare_classes_found)
        st.sidebar.metric("Total Classes", len(class_names))

        if len(class_names) == 0:
             st.error("No classes found in base or rare datasets. Check paths/contents.")
             st.stop()

        return combined_dataset, indices, class_names, num_base_classes

    except FileNotFoundError as e:
        st.error(f"Dataset path error: {e}. Check BASE_DATASET ('{base_path}') and RARE_DATASET ('{rare_path}').")
        st.stop()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        st.exception(e) # Show full traceback
        st.stop()

# --- Re-define transform globally if not done inside get_combined_dataset_and_indices ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Few-Shot Learning Functions ---

def create_episode(dataset, class_indices, class_list, n_way=5, n_shot=5, n_query=5):
    """Creates an episode for Prototypical Networks."""
    available_classes = list(class_indices.keys()) # Get labels (0, 1, 2...) that have samples
    if len(available_classes) < n_way:
        n_way = len(available_classes) # Adjust n_way if not enough classes
        if n_way < 2:
             # st.warning("Cannot create episode: < 2 classes available.") # Less verbose
             return None, None, None, None

    # Sample N-way CLASS LABELS from the available ones
    selected_class_ids = random.sample(available_classes, n_way)

    support_imgs, query_imgs = [], []
    support_labels, query_labels = [], []
    # Map original label (e.g., 3, 7, 1) to episode-specific label (0, 1, 2)
    episode_class_map = {original_label: episode_label for episode_label, original_label in enumerate(selected_class_ids)}
    actual_n_way = 0 # Track classes successfully added

    for original_label in selected_class_ids:
        indices_for_class = class_indices.get(original_label, [])

        min_samples_needed = n_shot + n_query
        if len(indices_for_class) < min_samples_needed:
            # Skip class for this episode if not enough samples
            continue

        # Sample indices FOR THIS CLASS from the dataset indices list
        sampled_indices = random.sample(indices_for_class, min_samples_needed)

        # Get images using the sampled dataset indices
        try:
            support_imgs += [dataset[i][0] for i in sampled_indices[:n_shot]]
            query_imgs += [dataset[i][0] for i in sampled_indices[n_shot:]]
        except IndexError as e:
             st.error(f"IndexError during episode creation for class {original_label}, index {i}. Check dataset/indices integrity.")
             st.exception(e)
             return None, None, None, None
        except Exception as e:
             st.error(f"Error retrieving data during episode creation: {e}")
             st.exception(e)
             return None, None, None, None

        # Assign new sequential labels (0 to n_way-1) for the episode
        new_label = episode_class_map[original_label]
        support_labels += [new_label] * n_shot
        query_labels += [new_label] * n_query
        actual_n_way += 1 # Increment count of classes successfully added

    if actual_n_way < 2: # Check if enough classes were actually added
        # st.warning(f"Episode creation resulted in < 2 valid classes ({actual_n_way}). Skipping.")
        return None, None, None, None

    # Return tensors on the correct device
    try:
        s_imgs_tensor = torch.stack(support_imgs).to(DEVICE)
        s_labels_tensor = torch.tensor(support_labels, dtype=torch.long).to(DEVICE)
        q_imgs_tensor = torch.stack(query_imgs).to(DEVICE)
        q_labels_tensor = torch.tensor(query_labels, dtype=torch.long).to(DEVICE)
        return s_imgs_tensor, s_labels_tensor, q_imgs_tensor, q_labels_tensor
    except Exception as e:
        st.error(f"Error stacking tensors in create_episode: {e}")
        st.exception(e)
        return None, None, None, None


def proto_loss(support_embeddings, support_labels, query_embeddings, query_labels):
    """Calculates the Prototypical Network loss and accuracy."""
    if support_embeddings is None or support_embeddings.numel() == 0 or \
       query_embeddings is None or query_embeddings.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0

    unique_episode_labels = torch.unique(support_labels)
    n_way_actual = len(unique_episode_labels)

    if n_way_actual < 2: # Need at least 2 classes for meaningful loss/accuracy
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0

    prototypes = []
    # Map the episode labels (e.g., 0, 1, 2...) to their prototype index (0, 1, 2...)
    proto_map = {label.item(): i for i, label in enumerate(unique_episode_labels)}

    for episode_label in unique_episode_labels:
        class_mask = (support_labels == episode_label)
        class_embeddings = support_embeddings[class_mask]
        if class_embeddings.size(0) > 0:
            prototypes.append(class_embeddings.mean(dim=0))
        else:
             st.warning(f"ProtoLoss: No support embeddings found for episode label {episode_label}. Skipping.")
             return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0

    if len(prototypes) != n_way_actual: # Should match unique labels count
        st.warning("ProtoLoss: Mismatch between unique labels and calculated prototypes.")
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0

    prototypes = torch.stack(prototypes) # Shape: [n_way_actual, embedding_dim]

    # Ensure query labels correspond to the classes present in the support set
    valid_query_mask = torch.isin(query_labels, unique_episode_labels)
    if not torch.any(valid_query_mask):
         # st.warning("ProtoLoss: No query samples match support set labels.")
         return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0 # No valid query samples

    filtered_query_embeddings = query_embeddings[valid_query_mask]
    filtered_query_labels_original = query_labels[valid_query_mask] # Keep original episode labels (e.g., 0, 1, 2...)

    # Calculate distances between filtered query embeddings and prototypes
    # Shape: [num_filtered_query, n_way_actual]
    distances = torch.cdist(filtered_query_embeddings, prototypes)

    # Get predictions based on nearest prototype (indices 0 to n_way_actual-1)
    predictions = torch.argmin(distances, dim=1) # Shape: [num_filtered_query]

    # Map the original filtered query labels to the prototype indices (0 to n_way_actual-1) for comparison
    mapped_true_labels = torch.tensor([proto_map[lbl.item()] for lbl in filtered_query_labels_original], dtype=torch.long).to(DEVICE)

    correct_predictions = (predictions == mapped_true_labels).sum().item()
    total_predictions = mapped_true_labels.size(0)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # Calculate cross-entropy loss using negative distances (closer = higher logit)
    loss = F.cross_entropy(-distances, mapped_true_labels)

    return loss, accuracy


# Recalculate prototypes after training for the entire dataset
@st.cache_data(show_spinner="Calculating final prototypes for all classes...") # Added spinner message
def calculate_final_prototypes(_model, _dataset, _class_names):
    _model.eval()
    all_embeddings = {} # Dict: label -> list of tensors

    # Use DataLoader for efficient batch processing
    # Consider pinning memory if using GPU and workers > 0
    loader = DataLoader(_dataset, batch_size=128, shuffle=False, num_workers=0)

    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            try:
                emb = _model(imgs)
                # Move embeddings to CPU *before* storing to avoid accumulating GPU memory
                emb_cpu = emb.cpu()
                labs_list = labs.tolist() # Convert labels tensor to list
                for i in range(emb_cpu.size(0)):
                    label = labs_list[i]
                    # Use setdefault for cleaner initialization
                    all_embeddings.setdefault(label, []).append(emb_cpu[i]) # Append CPU tensor
            except Exception as e:
                 st.error(f"Error during embedding calculation batch: {e}")
                 # Decide whether to continue or stop
                 continue # Continue with next batch

    final_prototypes = []
    prototype_labels = [] # Store the original dataset labels corresponding to prototypes
    unique_labels_present = sorted(list(all_embeddings.keys())) # Labels for which embeddings were found

    if not unique_labels_present:
        st.warning("No embeddings were generated. Cannot calculate prototypes.")
        return None, None

    for label in unique_labels_present:
        if not (0 <= label < len(_class_names)):
             st.warning(f"Skipping label {label} during prototype calculation: Out of bounds for class names list (len={len(_class_names)}).")
             continue

        class_embeddings_list = all_embeddings[label]
        if class_embeddings_list:
            try:
                class_embeddings = torch.stack(class_embeddings_list) # Stack CPU tensors
                prototype = class_embeddings.mean(dim=0) # Calculate mean on CPU
                final_prototypes.append(prototype)
                prototype_labels.append(label) # Append the original label
            except Exception as e:
                st.error(f"Error processing embeddings for class {label} ('{_class_names[label]}'): {e}")
                continue # Skip this class if stacking/mean fails
        # else: # Should not happen if label is in unique_labels_present keys
        #     st.warning(f"No embeddings found for class ID {label} ('{_class_names[label]}') though key existed.")

    if not final_prototypes:
         st.warning("Could not calculate any valid final prototypes.")
         return None, None

    # Stack final prototypes and move to target device
    final_prototypes_tensor = torch.stack(final_prototypes).to(DEVICE)
    st.success(f"Calculated {len(final_prototypes)} final prototypes for labels: {prototype_labels}")
    return final_prototypes_tensor, prototype_labels


# Visualize Prototypes Function
def visualize_prototypes(prototypes_tensor, prototype_labels, class_names_list):
    st.write("Visualizing Prototypes using PCA...")

    if prototypes_tensor is None or prototypes_tensor.numel() == 0:
        st.warning("⚠️ No prototypes available to visualize.")
        return

    num_prototypes = prototypes_tensor.size(0)
    if num_prototypes < 2:
        st.warning(f"⚠️ Need at least 2 prototypes for PCA. Found {num_prototypes}.")
        # Optionally display the single prototype info
        if num_prototypes == 1:
            st.write(f"Single prototype label: {prototype_labels[0]} ({class_names_list[prototype_labels[0]]})")
        return

    # Ensure number of labels matches number of prototypes
    if len(prototype_labels) != num_prototypes:
        st.error(f"Mismatch between number of prototypes ({num_prototypes}) and labels ({len(prototype_labels)}). Cannot visualize.")
        return

    pca = PCA(n_components=2)
    try:
        # Detach, move to CPU, convert to numpy
        prototypes_np = prototypes_tensor.detach().cpu().numpy()

        # Check for NaN or Inf before PCA
        if not np.all(np.isfinite(prototypes_np)):
            st.error("Prototypes contain NaN or Infinite values. Cannot perform PCA.")
            # Optionally show where NaNs are: np.isnan(prototypes_np).any(axis=1)
            return

        prototypes_2d = pca.fit_transform(prototypes_np)

        fig, ax = plt.subplots(figsize=(12, 9)) # Adjusted size

        # Use a colormap for potentially many classes
        cmap = plt.get_cmap('tab20', len(class_names_list)) # Use tab20 or other suitable map

        plotted_labels = set() # Keep track of labels already in legend

        for i, label_index in enumerate(prototype_labels):
             if 0 <= label_index < len(class_names_list):
                 class_name = class_names_list[label_index]
                 legend_label = f"{class_name} ({label_index})"
                 color = cmap(label_index / len(class_names_list)) # Assign color based on global index

                 # Only add label to legend once per class
                 if label_index not in plotted_labels:
                     ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1], label=legend_label, s=100, color=color, alpha=0.8)
                     plotted_labels.add(label_index)
                 else:
                    ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1], s=100, color=color, alpha=0.8) # No label if already added
             else:
                 # Handle labels out of bounds
                 ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1], label=f"Invalid Label ({label_index})", s=100, marker='x', color='red')
                 st.warning(f"Label index {label_index} out of bounds for class names list (length {len(class_names_list)}).")

        ax.set_title("Prototypes Visualization (PCA - 2 Components)")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        # Adjust legend position/size if too many classes
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside
        st.pyplot(fig)

    except ValueError as ve:
         st.error(f"PCA Error: {ve}. Check prototype dimensions and values.")
    except Exception as e:
        st.error(f"Error during PCA visualization: {e}")
        st.exception(e)


# --- Object Detection (YOLO) ---
@st.cache_resource(show_spinner="Loading detection model...")
def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO detection model from {YOLO_MODEL_PATH}: {e}")
        st.exception(e)
        st.stop()

# Detection function for YOLOv11
def detect_objects(image):
    model = load_yolo_model()

    # Convert PIL Image to NumPy array (OpenCV format BGR)
    img_array = np.array(image.convert("RGB")) # Ensure RGB first
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform inference
    try:
        results = model(img_bgr) # Run YOLO model
    except Exception as e:
        st.error(f"Error during YOLO inference: {e}")
        return img_array, pd.DataFrame() # Return original image and empty dataframe on error

    # Render results (draw bounding boxes on the image)
    # results[0].plot() returns a NumPy array (BGR)
    result_image_bgr = results[0].plot(conf=True, labels=True) # Show conf and labels on boxes

    # Convert result back to RGB for Streamlit display
    result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)

    # Extract detection details
    detections_list = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding boxes (xmin, ymin, xmax, ymax)
        confs = results[0].boxes.conf.cpu().numpy() # Confidence scores
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int) # Class IDs
        class_names = model.names # Get class names mapping from the model

        for i in range(len(boxes)):
             detections_list.append({
                 "Class": class_names.get(cls_ids[i], f"ID {cls_ids[i]}"), # Use .get for safety
                 "Confidence": confs[i],
                 "X_min": boxes[i, 0],
                 "Y_min": boxes[i, 1],
                 "X_max": boxes[i, 2],
                 "Y_max": boxes[i, 3],
             })

    detections_df = pd.DataFrame(detections_list)

    return result_image_rgb, detections_df


# === Main App Logic ===
st.title("🌿 Coffee Leaf Disease Classifier + Few-Shot Learning + Detection")

# --- Initialize Session State ---
# Use .setdefault() for cleaner initialization
st.session_state.setdefault('few_shot_trained', False)
st.session_state.setdefault('final_prototypes', None) # Tensor of prototypes
st.session_state.setdefault('prototype_labels', None) # List of original labels
st.session_state.setdefault('model_mode', 'standard') # 'standard' or 'few_shot'

# --- Load Data ---
# This runs once and caches results, or re-runs if cache is cleared or args change
# get_combined_dataset_and_indices handles errors internally and stops if needed
combined_dataset, class_indices, class_names, num_base_classes = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)


# --- Sidebar ---
st.sidebar.header("⚙️ Options & Status")

# --- Mode Selection / Status ---
st.sidebar.subheader("Mode")
# Button to switch back to standard classification
if st.sidebar.button("🔄 Reset to Standard Classifier"):
    st.session_state.model_mode = 'standard'
    st.session_state.few_shot_trained = False
    st.session_state.final_prototypes = None
    st.session_state.prototype_labels = None
    st.success("Switched to Standard Classification Mode.")
    # Clear cache related to few-shot results if necessary, e.g., prototype calculation cache
    # Note: calculate_final_prototypes uses @st.cache_data, might need manual clearing if model state changes matter
    st.cache_data.clear() # Clear data cache might be needed if dataset changed
    st.rerun()

# Display current mode
mode_status = "Standard Classifier"
if st.session_state.model_mode == 'few_shot' and st.session_state.final_prototypes is not None:
    mode_status = f"Few-Shot ({len(st.session_state.final_prototypes)} Prototypes Active)"
st.sidebar.info(f"**Current Mode:** {mode_status}")


# --- Load/Delete Saved Few-Shot Models ---
st.sidebar.divider()
st.sidebar.subheader("💾 Saved Few-Shot Models")

saved_model_names = list_saved_models()

# --- Loading Section ---
if not saved_model_names:
    st.sidebar.info("No saved few-shot models found.")
else:
    selected_model_to_load = st.sidebar.selectbox(
        "Load a saved few-shot state:",
        options=[""] + saved_model_names, # Add empty option for placeholder
        key="load_model_select",
        index=0 # Default to empty selection
    )
    if st.sidebar.button("📥 Load Selected State", key="load_model_button", disabled=(not selected_model_to_load)):
        if selected_model_to_load:
            # Get the current feature extractor instance to load weights into
            model_instance = cached_feature_extractor_model() # Get from cache
            # Get current class names for the crucial check during loading
            # Recalculate dataset info to be absolutely sure it's current
            _, _, current_cls_names_on_load, _ = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)
            if load_few_shot_state(selected_model_to_load, model_instance, current_cls_names_on_load):
                st.rerun() # Rerun to reflect loaded state in UI
        # No need for else, button is disabled if nothing selected

# --- Deleting Section ---
if saved_model_names: # Only show delete options if models exist
    st.sidebar.markdown("---") # Separator within the section
    selected_model_to_delete = st.sidebar.selectbox(
        "Delete a saved few-shot state:",
        options=[""] + saved_model_names,
        key="delete_model_select",
        index=0
    )
    if selected_model_to_delete: # Only show checkbox if a model is selected
        confirm_delete = st.sidebar.checkbox(f"Confirm deletion of '{selected_model_to_delete}'", key="delete_confirm")
        if st.sidebar.button("❌ Delete Selected State", key="delete_model_button", disabled=(not confirm_delete)):
             if confirm_delete: # Double check confirmation state
                 if delete_saved_model(selected_model_to_delete):
                     # Clear potentially cached list of models? list_saved_models isn't cached, so rerun is enough
                     st.rerun()
             # No need for else, button disabled if not confirmed
    else:
         st.sidebar.write("Select a model above to enable deletion.") # Guide user

else:
    # This case is covered by the "No saved models found" message above loading.
    pass


# --- Main Panel Options ---
option = st.radio(
    "Choose an action:",
    ["Upload & Predict", "Add/Manage Rare Classes", "Train Few-Shot Model", "Detection"],
    horizontal=True, key="main_option" # Added key for stability
)

# Select the appropriate model based on mode (done inside prediction logic)
# We load both cached models initially, ready for use
feature_extractor_model = cached_feature_extractor_model()
standard_classifier_model = cached_standard_classifier()


# --- Action Implementation ---

if option == "Upload & Predict":
    st.header("🔎 Upload Image for Prediction")
    uploaded_file = st.file_uploader("Choose a coffee leaf image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            # Display smaller image to save space
            st.image(image, caption="Uploaded Image", width=300) # Control width

            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            # Determine which model/method to use
            use_few_shot = (st.session_state.model_mode == 'few_shot' and
                            st.session_state.final_prototypes is not None and
                            st.session_state.prototype_labels is not None and
                            st.session_state.final_prototypes.numel() > 0 ) # Ensure prototypes are not empty

            if use_few_shot:
                # --- Few-Shot Prototype Prediction ---
                st.subheader("Prediction using Prototypes")
                model_to_use = feature_extractor_model # Use the feature extractor
                model_to_use.eval()
                with torch.no_grad():
                    embedding = model_to_use(input_tensor) # [1, embed_dim]
                    # Ensure prototypes are on the correct device and shape [num_prototypes, embed_dim]
                    prototypes_for_pred = st.session_state.final_prototypes.to(DEVICE)

                    # Calculate distances: [1, embed_dim] vs [num_prototypes, embed_dim] -> [1, num_prototypes]
                    distances = torch.cdist(embedding, prototypes_for_pred)
                    pred_prototype_index = torch.argmin(distances, dim=1).item() # Index within the prototype list

                    # Get the original dataset label corresponding to this prototype index
                    predicted_original_label = st.session_state.prototype_labels[pred_prototype_index]

                    # Map the original label to the class name
                    if 0 <= predicted_original_label < len(class_names):
                        predicted_class_name = class_names[predicted_original_label]
                        # Confidence calculation (using softmax on negative distances)
                        confidence_scores = torch.softmax(-distances, dim=1)
                        confidence = confidence_scores[0, pred_prototype_index].item()
                        st.metric(label="Prediction (Prototype)", value=predicted_class_name, delta=f"{confidence * 100:.1f}% Confidence")
                        st.info(f"(Matched prototype for class ID: {predicted_original_label})")
                    else:
                        st.error(f"Predicted prototype label index {predicted_original_label} is out of range for known class names ({len(class_names)}). Prototype labels might be inconsistent.")

            else:
                # --- Standard Classification Prediction ---
                st.subheader("Prediction using Standard Classifier")
                if st.session_state.model_mode != 'standard':
                    st.warning("Falling back to Standard Classifier mode (Few-shot prototypes not available or mode not set).")

                model_to_use = standard_classifier_model # Use the standard classifier
                model_to_use.eval()
                with torch.no_grad():
                    outputs = model_to_use(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_label = torch.argmax(probs, dim=1).item() # This label corresponds to base classes (0-4)
                    confidence = probs[0][pred_label].item()

                    # Map predicted label (0-4) to the corresponding class name from the base set
                    if 0 <= pred_label < num_base_classes: # Check against NUM_BASE_CLASSES
                        predicted_class_name = class_names[pred_label] # Get name from the combined list using the base index
                        st.metric(label="Prediction (Standard)", value=predicted_class_name, delta=f"{confidence * 100:.1f}% Confidence")
                    else:
                        # This case should technically not happen if the standard model only outputs 0-4
                        st.error(f"Standard classifier predicted label {pred_label}, which is out of range for base classes ({num_base_classes}). Model output issue?")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)


elif option == "Detection":
    st.header("🕵️ Object Detection with YOLO")
    uploaded_file_detect = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"], key="detect_uploader")

    if uploaded_file_detect:
        image_detect = Image.open(uploaded_file_detect).convert("RGB")
        result_image, detections = detect_objects(image_detect) # Calls the detection function

        # Resize for display (maintain aspect ratio) using Pillow ImageOps
        display_image = Image.fromarray(result_image) # Convert numpy array back to PIL Image
        display_image = ImageOps.contain(display_image, (900, 700)) # Resize while keeping aspect ratio within bounds

        st.image(display_image, caption="Detection Result", use_container_width=True) # Use column width

        # Show detection results
        if not detections.empty:
            st.subheader("📋 Detection Results:")
            # Format confidence as percentage
            detections['Confidence'] = detections['Confidence'].map('{:.1%}'.format)
            st.dataframe(detections[['Class', 'Confidence', 'X_min', 'Y_min', 'X_max', 'Y_max']]) # Display selected columns
        else:
            st.info("No objects detected.")


elif option == "Add/Manage Rare Classes":
    st.header("➕ Add New Rare Class (Few-Shot)")
    st.write(f"Upload exactly 5 sample images for the new disease class.")

    with st.form("add_class_form"):
        new_class_name = st.text_input("Enter the name for the new rare class:")
        uploaded_files_rare = st.file_uploader(
            "Upload 5 images:", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key="add_class_uploader"
        )
        submitted_add = st.form_submit_button("Add Class")

        if submitted_add:
            valid = True
            if not new_class_name:
                st.warning("Please enter a class name.")
                valid = False
            if len(uploaded_files_rare) != 5:
                st.warning(f"Please upload exactly 5 images. You uploaded {len(uploaded_files_rare)}.")
                valid = False

            if valid:
                # Sanitize class name for directory
                sanitized_class_name = "".join(c for c in new_class_name if c.isalnum() or c in (' ', '_')).strip().replace(" ", "_")
                if not sanitized_class_name:
                     st.error("Invalid class name after sanitization.")
                else:
                    new_class_dir = os.path.join(RARE_DATASET, sanitized_class_name)

                    if os.path.exists(new_class_dir):
                        st.warning(f"A class directory named '{sanitized_class_name}' already exists. Choose a different name or delete the existing one first.")
                    else:
                        try:
                            os.makedirs(new_class_dir, exist_ok=True)
                            image_save_errors = 0
                            for i, file in enumerate(uploaded_files_rare):
                                try:
                                    img = Image.open(file).convert("RGB")
                                    # Optional: Resize or standardize images on upload if needed
                                    # img = img.resize((256, 256))
                                    save_path = os.path.join(new_class_dir, f"sample_{i+1}.jpg")
                                    img.save(save_path, format='JPEG', quality=95) # Save as JPEG
                                except Exception as img_e:
                                    st.error(f"Error saving image {i+1} ({file.name}): {img_e}")
                                    image_save_errors += 1

                            if image_save_errors == 0:
                                st.success(f"✅ Added new class: '{sanitized_class_name}'. Please re-run 'Train Few-Shot Model' to incorporate it.")
                                # CRITICAL: Clear cache so dataset is reloaded with the new class
                                st.cache_data.clear()
                                # Clear potentially outdated prototypes if a class is added
                                st.session_state.final_prototypes = None
                                st.session_state.prototype_labels = None
                                st.session_state.few_shot_trained = False
                                st.session_state.model_mode = 'standard' # Reset to standard after adding class
                                st.rerun() # Force rerun to reload data
                            else:
                                st.error(f"Failed to save {image_save_errors} images. Class directory might be incomplete. Please check and try again.")
                                # Optionally remove the created directory if saving failed partially
                                # shutil.rmtree(new_class_dir)

                        except Exception as e:
                            st.error(f"Error creating directory or saving images for class '{sanitized_class_name}': {e}")
                            st.exception(e)

    st.divider()
    st.header("❌ Delete a Rare Class")

    try:
        # List only directories inside RARE_DATASET
        rare_class_dirs = [d for d in os.listdir(RARE_DATASET) if os.path.isdir(os.path.join(RARE_DATASET, d))]

        if not rare_class_dirs:
            st.info("No rare classes found to delete.")
        else:
            with st.form("delete_class_form"):
                to_delete = st.selectbox("Select rare class to delete:", rare_class_dirs, key="delete_rare_select")
                confirm_delete_rare = st.checkbox(f"Are you sure you want to permanently delete '{to_delete}' and its contents?", key="delete_rare_confirm")
                delete_submit_rare = st.form_submit_button("Delete Class")

                if delete_submit_rare:
                    if confirm_delete_rare:
                        delete_path = os.path.join(RARE_DATASET, to_delete)
                        try:
                            shutil.rmtree(delete_path)
                            st.success(f"✅ Deleted rare class: {to_delete}")
                            # CRITICAL: Clear cache and reset state
                            st.cache_data.clear()
                            st.session_state.few_shot_trained = False
                            st.session_state.final_prototypes = None
                            st.session_state.prototype_labels = None
                            st.session_state.model_mode = 'standard'
                            st.rerun() # Rerun to reload data and update UI
                        except Exception as e:
                            st.error(f"Error deleting directory {delete_path}: {e}")
                    else:
                        st.warning("Please confirm the deletion by checking the box.")

    except FileNotFoundError:
        st.info(f"Rare dataset directory '{RARE_DATASET}' not found or inaccessible.")
    except Exception as e:
        st.error(f"Error listing rare classes: {e}")


elif option == "Train Few-Shot Model":
    st.header("🚀 Train Few-Shot Model")
    # --- Check if enough classes exist before showing the form ---
    if len(class_names) < 2: # Need at least 2 classes for N-way=2 training
        st.error("Need at least two classes (Base + Rare combined) to perform few-shot training.")
        st.stop()
    else:
        st.info(f"Training will use all {len(class_names)} available classes: {class_names}")

    # --- Hardcoded Training Parameters ---
# --- Training Parameters ---
    epochs = 10
    # Set N-Way dynamically to the total number of available classes
    n_way_train = len(class_names)

    # Ensure N-way is at least 2 for meaningful training
    if n_way_train < 2:
        st.error(f"Training requires at least 2 classes, but found only {n_way_train}. Cannot proceed with few-shot training.")
        st.stop() # Stop if not enough classes

    episodes_per_epoch = 5 # Keep other parameters
    n_shot = 2
    n_query = 2
    learning_rate = 1e-5

    # Add a warning about potential resource usage if N-way is high
    # --- Training Form ---
    with st.form("train_form"):
        freeze_backbone = st.checkbox("❄️ Freeze Base Model Layers (Train only projection layer)", value=True, help="Recommended to prevent catastrophic forgetting of base classes.")
        submitted_train = st.form_submit_button("Start Training")

        if submitted_train:
            # Re-check class count just before training starts
            if len(class_names) < n_way_train:
                 st.error(f"Cannot start training. Need at least {n_way_train} classes for {n_way_train}-way training, found {len(class_names)}.")
                 st.stop()

            st.info("🚀 Starting few-shot training...")
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            chart_placeholder = st.empty() # Placeholder for the chart

            # --- Get model instance for training ---
            model_train = cached_feature_extractor_model() # Get the cached model
            model_train.train() # Set model to training mode

            # --- Optimizer Setup ---
            trainable_params = []
            if freeze_backbone:
                st.info("Freezing base model layers...")
                try:
                    # Freeze base model (assuming it's in model_train.model)
                    for param in model_train.model.parameters():
                        param.requires_grad = False
                    # Ensure projection layer is trainable (assuming model_train.projection)
                    for param in model_train.projection.parameters():
                        param.requires_grad = True
                    trainable_params = list(filter(lambda p: p.requires_grad, model_train.parameters()))
                    if not trainable_params:
                        st.error("No trainable parameters found (projection layer)! Check model structure ('model' and 'projection' attributes).")
                        st.stop()
                    st.success("Base layers frozen. Training projection layer only.")
                except AttributeError:
                    st.error("Could not access model.model or model.projection attributes to freeze/unfreeze. Training ALL layers.")
                    for param in model_train.parameters(): # Ensure all are trainable if fallback
                         param.requires_grad = True
                    trainable_params = list(model_train.parameters())
            else:
                st.info("Training all layers (base model + projection).")
                # Ensure all parameters are trainable
                for param in model_train.parameters():
                    param.requires_grad = True
                trainable_params = list(model_train.parameters())

            # Check if any trainable parameters were actually found
            if not trainable_params:
                 st.error("Optimizer setup failed: No trainable parameters collected.")
                 st.stop()

            optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-5)
           
            # --- Training Loop ---
            loss_history = []
            accuracy_history = []
            total_steps = epochs * episodes_per_epoch
            current_step = 0

            # Get fresh dataset info for episode creation within the loop if needed, or assume cached is fine
            # _, current_class_indices_train, current_class_names_train, _ = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)

            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                valid_episodes_in_epoch = 0

                for episode in range(episodes_per_epoch):
                    current_step += 1
                    # Create episode using current dataset state
                    s_imgs, s_labels, q_imgs, q_labels = create_episode(
                        combined_dataset, class_indices, class_names, n_way=n_way_train, n_shot=n_shot, n_query=n_query
                    )

                    if s_imgs is None or q_imgs is None: continue # Skip if episode creation failed

                    optimizer.zero_grad()
                    try:
                        s_emb = model_train(s_imgs)
                        q_emb = model_train(q_imgs)
                    except Exception as model_e:
                        st.error(f"Error during model forward pass in training (Epoch {epoch+1}, Ep {episode+1}): {model_e}")
                        st.exception(model_e)
                        continue # Skip episode on model error

                    loss, accuracy = proto_loss(s_emb, s_labels, q_emb, q_labels)

                    if loss is not None and not torch.isnan(loss) and loss.requires_grad:
                        try:
                            loss.backward()
                            # Optional: Gradient clipping
                            # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                            optimizer.step()
                            epoch_loss += loss.item()
                            epoch_accuracy += accuracy
                            valid_episodes_in_epoch += 1
                        except Exception as optim_e:
                             st.error(f"Error during optimizer step or backward pass (Epoch {epoch+1}, Ep {episode+1}): {optim_e}")
                             st.exception(optim_e)
                             # Consider stopping or just skipping step? Skipping for now.
                    # else: # Reduce verbosity for invalid loss skipping

                    # Update status and progress bar less frequently for performance
                    if (episode + 1) % 10 == 0 or episode == episodes_per_epoch - 1:
                        progress = current_step / total_steps
                        progress_bar.progress(progress)
                        status_placeholder.text(f"Epoch {epoch+1}/{epochs} | Episode {episode+1}/{episodes_per_epoch} | Current Loss: {loss.item():.4f} | Current Acc: {accuracy*100:.2f}%")


                # --- Log epoch results ---
                if valid_episodes_in_epoch > 0:
                     avg_loss = epoch_loss / valid_episodes_in_epoch
                     avg_accuracy = epoch_accuracy / valid_episodes_in_epoch
                     loss_history.append(avg_loss)
                     accuracy_history.append(avg_accuracy)
                     status_placeholder.text(f"Epoch {epoch+1}/{epochs} Completed - Avg Loss: {avg_loss:.4f} - Avg Accuracy: {avg_accuracy*100:.2f}%")
                else:
                      # Handle epochs where no valid episodes ran
                      loss_history.append(float('nan'))
                      accuracy_history.append(float('nan'))
                      status_placeholder.text(f"Epoch {epoch+1}/{epochs} Completed - No valid episodes were run.")


            status_placeholder.success("✅ Few-Shot Training Finished!")

            # --- Final Prototype Calculation (still inside 'if submitted') ---
            st.info("Calculating final prototypes...")
            # Ensure model is in eval mode
            model_train.eval()
            # Clear cache before calculating prototypes based on potentially new model state
            st.cache_data.clear() # Clear data cache
            # Recalculate dataset info to ensure it's current
            current_combined_dataset_proto, _, current_class_names_proto, _ = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)
            # Pass the *trained* model instance
            final_prototypes_tensor, final_prototype_labels = calculate_final_prototypes(model_train, current_combined_dataset_proto, current_class_names_proto)

            # --- Store results in session state ---
            if final_prototypes_tensor is not None:
                st.session_state.final_prototypes = final_prototypes_tensor
                st.session_state.prototype_labels = final_prototype_labels
                st.session_state.few_shot_trained = True
                st.session_state.model_mode = 'few_shot' # Switch mode automatically
                st.success("Prototypes Calculated. Model ready for Few-Shot Prediction.")

                # --- Display Training Curves (use the placeholder) ---
                chart_data = pd.DataFrame({
                    "Epoch": list(range(1, epochs + 1)),
                    "Average Loss": loss_history,
                    "Average Accuracy": accuracy_history
                })
                chart_placeholder.line_chart(chart_data.set_index("Epoch"))

            else:
                 # Ensure state is reset if prototype calculation fails
                 st.session_state.final_prototypes = None
                 st.session_state.prototype_labels = None
                 st.session_state.few_shot_trained = False
                 # Optionally reset mode? Or leave as is? Resetting is safer.
                 # st.session_state.model_mode = 'standard'
                 st.error("Failed to calculate final prototypes after training.")
                 chart_placeholder.empty() # Clear chart placeholder on failure

    # --- END OF TRAINING FORM `with st.form("train_form"):` BLOCK ---


    # --- SAVING SECTION (OUTSIDE AND AFTER THE FORM) ---
    # Show this section ONLY if few-shot training has run successfully (prototypes exist)
    # Check session state for prototypes to decide whether to show this.
    if st.session_state.get('final_prototypes') is not None and st.session_state.get('model_mode') == 'few_shot':
         st.divider()
         st.subheader("💾 Save Current Few-Shot State")
         st.info("Save the fine-tuned model weights and calculated prototypes.")
         # Use a different key for the text input to avoid conflicts
         save_model_name = st.text_input("Enter a name for this state:", key="save_model_name_input_main")

         # This button is now OUTSIDE the form, so it's allowed.
         if st.button("Save State", key="save_state_button_main"):
             if save_model_name:
                 # Need the model instance that was potentially trained.
                 # Getting it from cache should work if it was modified in-place.
                 model_to_save = cached_feature_extractor_model()
                 model_to_save.eval() # Ensure eval mode

                 # Get the current class names list for metadata
                 _, _, current_cls_names_for_saving, _ = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)

                 save_few_shot_state(
                     save_model_name,
                     model_to_save, # Pass the model instance
                     st.session_state.final_prototypes,
                     st.session_state.prototype_labels,
                     current_cls_names_for_saving
                 )
                 # Clear input field is tricky without callbacks, maybe omit for simplicity
             else:
                 st.warning("Please enter a name before saving.")




# --- Cleanup Temporary Directory (Optional) ---
# This might run on every script run, which is usually fine for temp dirs.
# Consider more robust cleanup if needed.
# try:
#     shutil.rmtree(TEMP_DIR)
# except Exception as e:
#     st.sidebar.warning(f"Could not cleanup temp dir {TEMP_DIR}: {e}")
##