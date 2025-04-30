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
# from sklearn.decomposition import PCA # PCA not used, removed
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
# TEMP_DIR = tempfile.mkdtemp() # Commented out as cleanup might cause issues

# --- Helper Functions for Saving/Loading Few-Shot States ---

def list_saved_models():
    """Returns a list of names of saved few-shot model states."""
    if not os.path.isdir(SAVED_MODELS_DIR):
        return []
    # List only directories within the SAVED_MODELS_DIR
    return [d for d in os.listdir(SAVED_MODELS_DIR) if os.path.isdir(os.path.join(SAVED_MODELS_DIR, d))]

def save_few_shot_state(name, model, prototypes, proto_labels, current_class_names, few_shot_strategy):
    """Saves the model state, prototypes, strategy, and metadata."""
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
    # Use columns to place button beside warning
    proceed_with_save = True
    if os.path.exists(save_dir):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning(f"Model name '{sanitized_name}' already exists.")
        with col2:
            # Generate a unique key based on the name for the button
            overwrite_key = f"overwrite_button_{sanitized_name}"
            if not st.button("Overwrite?", key=overwrite_key):
                st.info("Save cancelled. Choose a different name or click 'Overwrite?'.")
                proceed_with_save = False # Don't proceed if overwrite not clicked
            else:
                 st.info(f"Overwriting '{sanitized_name}'...") # Info if overwrite is clicked
    else:
         st.info(f"Saving new model state '{sanitized_name}'...")


    if not proceed_with_save:
        return False

    try:
        os.makedirs(save_dir, exist_ok=True) # Create directory

        # 1. Save Model State Dictionary
        model.to('cpu') # Move model to CPU
        model_path = os.path.join(save_dir, "feature_extractor_state_dict.pth")
        torch.save(model.state_dict(), model_path)
        model.to(DEVICE) # Move model back to original device

        # 2. Save Prototypes Tensor
        prototypes_path = os.path.join(save_dir, "prototypes.pt")
        torch.save(prototypes.cpu(), prototypes_path)

        # 3. Save Metadata
        if few_shot_strategy != 'train_projection':
            st.error(f"Internal Error: Attempting to save with invalid strategy '{few_shot_strategy}'. Expected 'train_projection'. Save cancelled.")
            if os.path.exists(save_dir): shutil.rmtree(save_dir)
            return False

        metadata = {
            "prototype_labels": proto_labels, # Standard list
            "class_names_on_save": current_class_names, # List of strings
            "few_shot_strategy": 'train_projection' # Hardcoded
        }

        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        st.success(f"Few-shot model state '{sanitized_name}' saved successfully!")
        return True
    except Exception as e:
        st.error(f"Error saving model state '{sanitized_name}': {e}")
        st.exception(e)
        if os.path.exists(save_dir):
             try:
                 shutil.rmtree(save_dir)
                 st.info(f"Cleaned up partially saved directory '{save_dir}'.")
             except Exception as cleanup_e:
                 st.error(f"Error cleaning up directory during save failure: {cleanup_e}")
        return False

def load_few_shot_state(name, model_to_load_into, current_class_names):
    """Loads a saved model state, prototypes, labels, and strategy into session state and the model."""
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
        loaded_strategy = metadata.get("few_shot_strategy")

        if loaded_proto_labels is None or saved_class_names is None or loaded_strategy is None:
             st.error(f"Metadata file for '{name}' is corrupted or missing required keys (labels, class_names, strategy).")
             return False

        if loaded_strategy != 'train_projection':
            st.error(f"Saved model '{name}' used strategy '{loaded_strategy}', but only 'train_projection' (frozen backbone) is currently supported. Cannot load.")
            return False

        # **CRUCIAL CHECK**: Compare saved class names with current class names
        if set(saved_class_names) != set(current_class_names):
            st.warning(f"⚠️ **Class Mismatch!**")
            st.warning(f"Saved model '{name}' classes: `{saved_class_names}`")
            st.warning(f"Current active classes: `{current_class_names}`")
            st.warning("Predictions might be incorrect or errors may occur. Proceed with caution.")

        # 2. Load Model State Dictionary
        model_to_load_into.to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        try:
            missing_keys, unexpected_keys = model_to_load_into.load_state_dict(state_dict, strict=True)
            if missing_keys: st.warning(f"Loaded state dict is missing keys: {missing_keys}")
            if unexpected_keys: st.warning(f"Loaded state dict has unexpected keys: {unexpected_keys}")
        except RuntimeError as e:
            st.error(f"RuntimeError loading state_dict for '{name}'. Architecture mismatch? {e}")
            st.error("This usually means the saved model structure (base + projection) doesn't match the current code's structure.")
            return False
        model_to_load_into.eval()

        # 3. Load Prototypes
        loaded_prototypes = torch.load(prototypes_path, map_location=DEVICE)

        # 4. Update Session State
        st.session_state.final_prototypes = loaded_prototypes
        st.session_state.prototype_labels = loaded_proto_labels
        st.session_state.few_shot_strategy = loaded_strategy
        st.session_state.few_shot_trained = True
        st.session_state.model_mode = 'few_shot'

        st.success(f"Successfully loaded few-shot model state '{name}' (Strategy: {loaded_strategy}). Mode set to Few-Shot.")
        return True

    except Exception as e:
        st.error(f"Error loading model state '{name}': {e}")
        st.exception(e)
        # Reset state if loading fails partially
        st.session_state.final_prototypes = None
        st.session_state.prototype_labels = None
        st.session_state.few_shot_strategy = None
        st.session_state.few_shot_trained = False
        st.session_state.model_mode = 'standard'
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

class EfficientNetWithProjection(nn.Module):
    def __init__(self, base_model, output_dim=1024):
        super(EfficientNetWithProjection, self).__init__()
        self.model = base_model
        in_features = 1280
        self.projection = nn.Linear(in_features, output_dim)

    def forward(self, x):
        features = self.model(x)
        projected_features = self.projection(features)
        return projected_features

def get_base_efficientnet_architecture(num_classes=5):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_feature_extractor_base():
    base_model = get_base_efficientnet_architecture(num_classes=5)
    try:
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
        missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
        if unexpected_keys and not all(k.startswith('classifier.') for k in unexpected_keys):
             st.warning(f"Loading base weights: Unexpected keys found beyond classifier: {unexpected_keys}")
        if missing_keys:
             st.warning(f"Loading base weights: Missing keys: {missing_keys}")
    except Exception as e:
        st.error(f"Error loading model weights from {MODEL_WEIGHTS_PATH} into base architecture: {e}")
        st.exception(e)
        st.stop()
    base_model.classifier = nn.Identity()
    base_model.eval()
    return base_model

def load_standard_classifier():
    model = get_base_efficientnet_architecture(num_classes=5)
    try:
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        st.error(f"Error loading model weights for standard classifier: {e}")
        st.exception(e)
        st.stop()
    model.to(DEVICE)
    model.eval()
    return model


# --- Caching ---
@st.cache_resource
def cached_feature_extractor_model():
    base_model = get_feature_extractor_base()
    model = EfficientNetWithProjection(base_model, output_dim=1024)
    model.to(DEVICE)
    model.eval()
    st.sidebar.info("Feature extractor model ready (cached).")
    return model

@st.cache_resource
def cached_standard_classifier():
    model = load_standard_classifier()
    st.sidebar.info("Standard classifier model ready (cached).")
    return model

@st.cache_data
def get_combined_dataset_and_indices(base_path, rare_path):
    try:
        transform_local = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        full_dataset = datasets.ImageFolder(base_path, transform_local)
        num_base_classes = len(full_dataset.classes)
        base_class_names = sorted(full_dataset.classes)
        rare_classes_found = 0
        rare_class_names = []
        if os.path.isdir(rare_path) and any(os.scandir(rare_path)):
            try:
                rare_dataset = datasets.ImageFolder(rare_path, transform_local)
                if len(rare_dataset.samples) > 0:
                    rare_dataset.samples = [(path, label + num_base_classes) for path, label in rare_dataset.samples]
                    combined_dataset = ConcatDataset([full_dataset, rare_dataset])
                    rare_classes_found = len(rare_dataset.classes)
                    rare_class_names = sorted(rare_dataset.classes)
                else:
                    combined_dataset = full_dataset
            except Exception as e_rare:
                st.warning(f"Could not load rare dataset from {rare_path}: {e_rare}. Using base dataset only.")
                combined_dataset = full_dataset
        else:
            combined_dataset = full_dataset
        indices = {}
        current_idx = 0
        if isinstance(combined_dataset, ConcatDataset):
            for ds in combined_dataset.datasets:
                 if hasattr(ds, 'samples'):
                      for _, label in ds.samples:
                           indices.setdefault(label, []).append(current_idx)
                           current_idx += 1
                 else:
                      for i in range(len(ds)):
                           _, label = ds[i]
                           indices.setdefault(label, []).append(current_idx)
                           current_idx += 1
        elif isinstance(combined_dataset, datasets.ImageFolder):
             for idx, (_, label) in enumerate(combined_dataset.samples):
                  indices.setdefault(label, []).append(idx)
        else:
            st.error("Unexpected dataset type encountered when building indices.")
            st.stop()
        class_names = base_class_names + rare_class_names
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
        st.exception(e)
        st.stop()

# --- Global transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Few-Shot Learning Functions ---

def create_episode(dataset, class_indices, class_list, n_way=5, n_shot=5, n_query=5):
    """Creates an episode for Prototypical Networks."""
    available_classes = list(class_indices.keys())
    if len(available_classes) < n_way:
        n_way = len(available_classes)
        if n_way < 2:
            st.error(f"Episode creation failed: Need at least 2 classes with enough samples, found {n_way}.")
            return None, None, None, None
    eligible_classes = [
        cls_id for cls_id in available_classes
        if len(class_indices.get(cls_id, [])) >= (n_shot + n_query)
    ]
    if len(eligible_classes) < n_way:
        n_way = len(eligible_classes)
        if n_way < 2:
            st.error(f"Episode creation failed: Need at least 2 eligible classes (with {n_shot+n_query} samples each). Found {n_way}.")
            return None, None, None, None
    selected_class_ids = random.sample(eligible_classes, n_way)
    support_imgs, query_imgs = [], []
    support_labels, query_labels = [], []
    episode_class_map = {original_label: episode_label for episode_label, original_label in enumerate(selected_class_ids)}
    for original_label in selected_class_ids:
        indices_for_class = class_indices.get(original_label, [])
        sampled_indices = random.sample(indices_for_class, n_shot + n_query)
        try:
            # Check if dataset is ConcatDataset or ImageFolder to access items correctly
            if isinstance(dataset, ConcatDataset):
                support_imgs += [dataset[i][0] for i in sampled_indices[:n_shot]]
                query_imgs += [dataset[i][0] for i in sampled_indices[n_shot:]]
            elif isinstance(dataset, datasets.ImageFolder):
                 # Direct access might be okay, but let's be safe
                 support_imgs += [dataset[i][0] for i in sampled_indices[:n_shot]]
                 query_imgs += [dataset[i][0] for i in sampled_indices[n_shot:]]
            else:
                 st.error("Unhandled dataset type in create_episode access.")
                 return None, None, None, None
        except IndexError as e:
            # More specific error message
            st.error(f"IndexError during episode creation. Class: {original_label}, Attempted Index: {i} (from sampled: {sampled_indices}). Total items in dataset: {len(dataset)}. Check dataset/indices integrity.")
            st.exception(e)
            return None, None, None, None
        except Exception as e:
            st.error(f"Error retrieving data during episode creation: {e}")
            st.exception(e)
            return None, None, None, None
        new_label = episode_class_map[original_label]
        support_labels += [new_label] * n_shot
        query_labels += [new_label] * n_query
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
    if n_way_actual < 2:
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0
    prototypes = []
    for episode_label in range(n_way_actual):
        class_mask = (support_labels == episode_label)
        if torch.any(class_mask):
             class_embeddings = support_embeddings[class_mask]
             prototypes.append(class_embeddings.mean(dim=0))
        else:
            st.warning(f"ProtoLoss: No support embeddings found for episode label {episode_label}. Skipping prototype.")
            return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0
    if len(prototypes) != n_way_actual:
        st.warning(f"ProtoLoss: Mismatch between expected ways ({n_way_actual}) and calculated prototypes ({len(prototypes)}).")
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0
    prototypes = torch.stack(prototypes)
    valid_query_mask = torch.isin(query_labels, unique_episode_labels)
    if not torch.any(valid_query_mask):
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0
    filtered_query_embeddings = query_embeddings[valid_query_mask]
    filtered_query_labels = query_labels[valid_query_mask]
    distances = torch.cdist(filtered_query_embeddings, prototypes)
    predictions = torch.argmin(distances, dim=1)
    correct_predictions = (predictions == filtered_query_labels).sum().item()
    total_predictions = filtered_query_labels.size(0)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    loss = F.cross_entropy(-distances, filtered_query_labels)
    return loss, accuracy


@st.cache_data(show_spinner="Calculating final prototypes for all classes...")
def calculate_final_prototypes(_model, _dataset, _class_names, _strategy):
    if _strategy != 'train_projection':
         st.warning(f"calculate_final_prototypes called with unexpected strategy: '{_strategy}'. Proceeding as if 'train_projection'.")
    _model.eval()
    all_embeddings = {}
    # Use num_workers=0 for Windows compatibility in Streamlit Cloud/some environments
    loader = DataLoader(_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True if DEVICE=='cuda' else False)
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            try:
                emb = _model(imgs)
                emb_cpu = emb.cpu()
                labs_list = labs.tolist()
                for i in range(emb_cpu.size(0)):
                    label = labs_list[i]
                    all_embeddings.setdefault(label, []).append(emb_cpu[i])
            except Exception as e:
                st.error(f"Error during embedding calculation batch: {e}")
                continue
    final_prototypes = []
    prototype_labels = []
    unique_labels_present = sorted(list(all_embeddings.keys()))
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
                class_embeddings = torch.stack(class_embeddings_list)
                prototype = class_embeddings.mean(dim=0)
                final_prototypes.append(prototype)
                prototype_labels.append(label)
            except Exception as e:
                st.error(f"Error processing embeddings for class {label} ('{_class_names[label]}'): {e}")
                continue
    if not final_prototypes:
        st.warning("Could not calculate any valid final prototypes.")
        return None, None
    final_prototypes_tensor = torch.stack(final_prototypes).to(DEVICE)
    st.success(f"Calculated {len(final_prototypes)} final prototypes (Strategy: {_strategy}) for original labels: {prototype_labels}")
    return final_prototypes_tensor, prototype_labels


# --- Object Detection (YOLO) ---
@st.cache_resource(show_spinner="Loading detection model...")
def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO detection model from {YOLO_MODEL_PATH}: {e}")
        st.exception(e)
        st.stop()

def detect_objects(image):
    model = load_yolo_model()
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    try:
        results = model(img_bgr, device=DEVICE)
    except Exception as e:
        st.error(f"Error during YOLO inference: {e}")
        return img_array, pd.DataFrame()
    result_image_bgr = results[0].plot(conf=True, labels=True)
    result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)
    detections_list = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = model.names
        for i in range(len(boxes)):
            detections_list.append({
                "Class": class_names.get(cls_ids[i], f"ID {cls_ids[i]}"),
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
st.session_state.setdefault('few_shot_trained', False)
st.session_state.setdefault('final_prototypes', None)
st.session_state.setdefault('prototype_labels', None)
st.session_state.setdefault('model_mode', 'standard')
st.session_state.setdefault('few_shot_strategy', None)

# --- Load Data ---
combined_dataset, class_indices, class_names, num_base_classes = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)

# --- Sidebar ---
st.sidebar.header("⚙️ Options & Status")

# --- Mode Selection / Status ---
st.sidebar.subheader("Mode")
if st.sidebar.button("🔄 Reset to Standard Classifier"):
    st.session_state.model_mode = 'standard'
    st.session_state.few_shot_trained = False
    st.session_state.final_prototypes = None
    st.session_state.prototype_labels = None
    st.session_state.few_shot_strategy = None
    st.success("Switched to Standard Classification Mode.")
    st.cache_data.clear()
    st.cache_resource.clear() # Also clear resource cache if needed
    st.rerun()

mode_status = "Standard Classifier"
strategy_info = ""
if st.session_state.model_mode == 'few_shot' and st.session_state.final_prototypes is not None:
    mode_status = f"Few-Shot ({len(st.session_state.final_prototypes)} Prototypes)"
    strategy_info = f"(Strategy: {st.session_state.get('few_shot_strategy', 'N/A').replace('_', ' ').title()})"
st.sidebar.info(f"**Current Mode:** {mode_status} {strategy_info}")


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
        options=[""] + saved_model_names,
        key="load_model_select",
        index=0
    )
    if st.sidebar.button("📥 Load Selected State", key="load_model_button", disabled=(not selected_model_to_load)):
        if selected_model_to_load:
            model_instance = cached_feature_extractor_model()
            # Need current class names for the check during load
            _, _, current_cls_names_on_load, _ = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)
            if load_few_shot_state(selected_model_to_load, model_instance, current_cls_names_on_load):
                st.rerun()

# --- Deleting Section ---
if saved_model_names:
    st.sidebar.markdown("---")
    selected_model_to_delete = st.sidebar.selectbox(
        "Delete a saved few-shot state:",
        options=[""] + saved_model_names,
        key="delete_model_select",
        index=0
    )
    if selected_model_to_delete:
        confirm_delete = st.sidebar.checkbox(f"Confirm deletion of '{selected_model_to_delete}'", key="delete_confirm")
        if st.sidebar.button("❌ Delete Selected State", key="delete_model_button", disabled=(not confirm_delete)):
             if confirm_delete:
                 if delete_saved_model(selected_model_to_delete):
                     st.rerun()


# --- Main Panel Options ---
option = st.radio(
    "Choose an action:",
    ["Upload & Predict", "Add/Manage Rare Classes", "Train Few-Shot Model", "Detection"],
    horizontal=True, key="main_option"
)

# Load models (cached)
feature_extractor_model = cached_feature_extractor_model()
standard_classifier_model = cached_standard_classifier()


# --- Action Implementation ---

if option == "Upload & Predict":
    st.header("🔎 Upload Image for Prediction")
    uploaded_file = st.file_uploader("Choose a coffee leaf image...", type=["jpg", "jpeg", "png"], key="file_uploader")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=300)
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            use_few_shot = (st.session_state.model_mode == 'few_shot' and
                            st.session_state.final_prototypes is not None and
                            st.session_state.prototype_labels is not None and
                            st.session_state.few_shot_strategy == 'train_projection' and
                            st.session_state.final_prototypes.numel() > 0 )
            if use_few_shot:
                st.subheader("Prediction using Prototypes")
                model_to_use = feature_extractor_model
                model_to_use.eval()
                strategy_for_pred = st.session_state.few_shot_strategy
                with torch.no_grad():
                    embedding = model_to_use(input_tensor)
                    prototypes_for_pred = st.session_state.final_prototypes.to(DEVICE)
                    if embedding.shape[1] != prototypes_for_pred.shape[1]:
                        st.error(f"Dimension mismatch! Embedding dim: {embedding.shape[1]}, Prototype dim: {prototypes_for_pred.shape[1]}.")
                        st.stop()
                    distances = torch.cdist(embedding, prototypes_for_pred)
                    pred_prototype_index = torch.argmin(distances, dim=1).item()
                    predicted_original_label = st.session_state.prototype_labels[pred_prototype_index]
                    if 0 <= predicted_original_label < len(class_names):
                        predicted_class_name = class_names[predicted_original_label]
                        confidence_scores = torch.softmax(-distances, dim=1)
                        confidence = confidence_scores[0, pred_prototype_index].item()
                        st.metric(label="Prediction (Prototype)", value=predicted_class_name, delta=f"{confidence * 100:.1f}% Confidence")
                    else:
                        st.error(f"Predicted prototype label index {predicted_original_label} is out of range for known class names ({len(class_names)}).")
            else:
                st.subheader("Prediction using Standard Classifier")
                if st.session_state.model_mode != 'standard':
                    st.warning("Falling back to Standard Classifier mode (Few-shot prototypes not available or invalid state).")
                model_to_use = standard_classifier_model
                model_to_use.eval()
                with torch.no_grad():
                    outputs = model_to_use(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_label = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_label].item()
                    if 0 <= pred_label < num_base_classes:
                        predicted_class_name = class_names[pred_label]
                        st.metric(label="Prediction (Standard)", value=predicted_class_name, delta=f"{confidence * 100:.1f}% Confidence")
                    else:
                        st.error(f"Standard classifier predicted label {pred_label}, which is out of range for base classes ({num_base_classes}).")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)

elif option == "Detection":
    st.header("🕵️ Object Detection with YOLO")
    uploaded_file_detect = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"], key="detect_uploader")
    if uploaded_file_detect:
        try:
            image_detect = Image.open(uploaded_file_detect).convert("RGB")
            result_image, detections = detect_objects(image_detect)
            display_image = Image.fromarray(result_image)
            # Use contain for better resizing within limits
            display_image = ImageOps.contain(display_image, (900, 700))
            st.image(display_image, caption="Detection Result", use_container_width=True)
            if not detections.empty:
                st.subheader("📋 Detection Results:")
                detections['Confidence'] = detections['Confidence'].map('{:.1%}'.format)
                detections[['X_min', 'Y_min', 'X_max', 'Y_max']] = detections[['X_min', 'Y_min', 'X_max', 'Y_max']].round(1)
                st.dataframe(detections[['Class', 'Confidence', 'X_min', 'Y_min', 'X_max', 'Y_max']])
            else:
                st.info("No objects detected.")
        except Exception as e:
             st.error(f"An error occurred during detection: {e}")
             st.exception(e)

elif option == "Add/Manage Rare Classes":
    st.header("➕ Add New Rare Class (Few-Shot)")
    # Requirement should match n_shot + n_query used in training
    n_shot_req = 2 # From training params
    n_query_req = 2 # From training params
    required_samples = n_shot_req + n_query_req
    st.write(f"Upload at least **{required_samples}** sample images for the new disease class (for training stability).")

    with st.form("add_class_form"):
        new_class_name = st.text_input("Enter the name for the new rare class:")
        uploaded_files_rare = st.file_uploader(
            f"Upload {required_samples} or more images:", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key="add_class_uploader"
        )
        submitted_add = st.form_submit_button("Add Class")
        if submitted_add:
            valid = True
            if not new_class_name or not new_class_name.strip():
                st.warning("Please enter a valid class name.")
                valid = False
            if len(uploaded_files_rare) < required_samples:
                st.warning(f"Please upload at least {required_samples} images. You uploaded {len(uploaded_files_rare)}.")
                valid = False
            if valid:
                sanitized_class_name = "".join(c for c in new_class_name if c.isalnum() or c in (' ', '_')).strip().replace(" ", "_")
                if not sanitized_class_name:
                     st.error("Invalid class name after sanitization (contains only spaces or invalid chars).")
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
                                    # Improve filename uniqueness
                                    base, ext = os.path.splitext(file.name)
                                    safe_base = "".join(c for c in base if c.isalnum() or c in ('_', '-')).strip()[:50] # Sanitize and limit length
                                    filename = f"{safe_base}_{random.randint(1000, 9999)}_{i+1}.jpg"
                                    save_path = os.path.join(new_class_dir, filename)
                                    img.save(save_path, format='JPEG', quality=95)
                                except Exception as img_e:
                                    st.error(f"Error saving image {i+1} ({file.name}): {img_e}")
                                    image_save_errors += 1
                            if image_save_errors == 0:
                                st.success(f"✅ Added new class: '{sanitized_class_name}' with {len(uploaded_files_rare)} images. Please re-run 'Train Few-Shot Model' to incorporate it.")
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                st.session_state.final_prototypes = None
                                st.session_state.prototype_labels = None
                                st.session_state.few_shot_strategy = None
                                st.session_state.few_shot_trained = False
                                st.session_state.model_mode = 'standard'
                                st.rerun()
                            else:
                                st.error(f"Failed to save {image_save_errors} images. Class directory might be incomplete. Please check and try again.")
                        except Exception as e:
                            st.error(f"Error creating directory or saving images for class '{sanitized_class_name}': {e}")
                            st.exception(e)

    st.divider()
    st.header("❌ Delete a Rare Class")
    try:
        if os.path.isdir(RARE_DATASET):
            rare_class_dirs = [d for d in os.listdir(RARE_DATASET) if os.path.isdir(os.path.join(RARE_DATASET, d))]
            if not rare_class_dirs:
                st.info("No rare classes found to delete.")
            else:
                with st.form("delete_class_form"):
                    to_delete = st.selectbox("Select rare class to delete:", rare_class_dirs, key="delete_rare_select")
                    confirm_delete_rare = st.checkbox(f"Are you sure you want to permanently delete '{to_delete}' and its contents?", key="delete_rare_confirm")
                    delete_submit_rare = st.form_submit_button("Delete Class")
                    if delete_submit_rare:
                        if confirm_delete_rare and to_delete:
                            delete_path = os.path.join(RARE_DATASET, to_delete)
                            try:
                                shutil.rmtree(delete_path)
                                st.success(f"✅ Deleted rare class: {to_delete}")
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                st.session_state.few_shot_trained = False
                                st.session_state.final_prototypes = None
                                st.session_state.prototype_labels = None
                                st.session_state.few_shot_strategy = None
                                st.session_state.model_mode = 'standard'
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting directory {delete_path}: {e}")
                        elif not confirm_delete_rare:
                            st.warning("Please confirm the deletion by checking the box.")
                        # else: # Should not happen if selectbox has options
                        #    st.warning("Please select a class to delete.")
        else:
             st.info(f"Rare dataset directory '{RARE_DATASET}' does not exist.")
    except Exception as e:
        st.error(f"Error listing or deleting rare classes: {e}")
        st.exception(e)


elif option == "Train Few-Shot Model":
    st.header("🚀 Train Few-Shot Model")
    if len(class_names) < 2:
        st.error("Need at least two classes (Base + Rare combined) to perform few-shot training.")
        st.stop()

    # --- Training Parameters ---
    epochs = 10
    n_way_train = len(class_names) # Use all available classes
    episodes_per_epoch = 10
    n_shot = 2
    n_query = 2
    learning_rate = 1e-4
    weight_decay_proj = 1e-4

    # Check if enough eligible classes exist before showing form
    eligible_classes_check = [
        cls_id for cls_id in class_indices
        if len(class_indices.get(cls_id, [])) >= (n_shot + n_query)
    ]
    if len(eligible_classes_check) < 2:
         st.error(f"Cannot start training. Need at least 2 classes with enough samples ({n_shot + n_query} each). Found {len(eligible_classes_check)} eligible classes: {[class_names[i] for i in eligible_classes_check]}.")
         st.stop()


    # --- Training Form ---
    with st.form("train_form"):
        
        submitted_train = st.form_submit_button("Start Few-Shot Training")

        if submitted_train:
            active_strategy = 'train_projection' # Hardcoded strategy
            st.info(f"🚀 Starting few-shot process")

            # Re-check class count just before training starts (might have changed)
            _, current_indices, current_names, _ = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)
            current_n_way = len(current_names)
            if current_n_way < 2:
                st.error(f"Cannot start training. Need at least 2 classes, found {current_n_way}.")
                st.stop()
            # Check eligible classes again
            current_eligible = [
                cls_id for cls_id in current_indices
                if len(current_indices.get(cls_id, [])) >= (n_shot + n_query)
            ]
            if len(current_eligible) < 2:
                 st.error(f"Cannot start training. Need at least 2 classes with ({n_shot + n_query}) samples. Found {len(current_eligible)} eligible.")
                 st.stop()

            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            chart_placeholder = st.empty()

            model_train = cached_feature_extractor_model()
            model_train.train()

            # --- Optimizer Setup ---
            trainable_params = []
            try:
                for param in model_train.model.parameters(): # Freeze base
                    param.requires_grad = False
                for param in model_train.projection.parameters(): # Unfreeze projection
                    param.requires_grad = True
                trainable_params = list(filter(lambda p: p.requires_grad, model_train.parameters()))
                if not trainable_params:
                    st.error("No trainable parameters found in projection layer!")
                    st.stop()
            except AttributeError as e:
                st.error(f"Model structure error: {e}")
                st.stop()

            optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay_proj)

            # --- Training Loop ---
            loss_history = []
            accuracy_history = []
            total_steps = epochs * episodes_per_epoch
            current_step = 0
            training_successful = False # Flag to track success

            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                valid_episodes_in_epoch = 0
                for episode in range(episodes_per_epoch):
                    current_step += 1
                    # Use current dataset/indices/names for episode creation
                    s_imgs, s_labels, q_imgs, q_labels = create_episode(
                        combined_dataset, current_indices, current_names, n_way=current_n_way, n_shot=n_shot, n_query=n_query
                    )
                    if s_imgs is None or q_imgs is None:
                        st.warning(f"Skipping episode {episode+1}/{episodes_per_epoch} in epoch {epoch+1} (creation failed).")
                        continue
                    try:
                        s_emb = model_train(s_imgs)
                        q_emb = model_train(q_imgs)
                        loss, accuracy = proto_loss(s_emb, s_labels, q_emb, q_labels)
                    except Exception as model_e:
                        st.error(f"Error in model/loss (Ep {epoch+1}-{episode+1}): {model_e}")
                        continue
                    if loss is not None and not torch.isnan(loss) and loss.requires_grad:
                        try:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                            epoch_accuracy += accuracy
                            valid_episodes_in_epoch += 1
                        except Exception as optim_e:
                            st.error(f"Error during optimizer step (Ep {epoch+1}-{episode+1}): {optim_e}")
                            # Optionally break or continue cautiously
                    elif loss is not None and not torch.isnan(loss):
                         st.warning(f"Loss does not require grad (Ep {epoch+1}-{episode+1}). Step skipped.")
                         epoch_loss += loss.item() # Still record loss
                         epoch_accuracy += accuracy
                         valid_episodes_in_epoch += 1

                    # Update progress less frequently
                    if (episode + 1) % 5 == 0 or episode == episodes_per_epoch - 1:
                        progress = current_step / total_steps
                        progress_bar.progress(min(progress, 1.0))

                # Log epoch results
                

            status_placeholder.success("✅ Few-Shot Training Process Finished!")

            # --- Final Prototype Calculation ---
        
            model_train.eval()
            st.cache_data.clear() # Clear before calculating prototypes
            current_combined_dataset_proto, _, current_class_names_proto, _ = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)
            final_prototypes_tensor, final_prototype_labels = calculate_final_prototypes(
                model_train, current_combined_dataset_proto, current_class_names_proto, active_strategy
            )

            # --- Store results in session state ---
            if final_prototypes_tensor is not None and final_prototype_labels is not None:
                st.session_state.final_prototypes = final_prototypes_tensor
                st.session_state.prototype_labels = final_prototype_labels
                st.session_state.few_shot_strategy = active_strategy
                st.session_state.few_shot_trained = True
                st.session_state.model_mode = 'few_shot'
                st.success(f"Prototypes Calculated. Model ready for Few-Shot Prediction.")
                training_successful = True # Mark as successful

                # Optional: Plotting (can be slow)
                # chart_data = pd.DataFrame({"Epoch": list(range(1, epochs + 1)), "Avg Loss": loss_history, "Avg Accuracy": accuracy_history}).dropna()
                # if not chart_data.empty:
                #     chart_placeholder.line_chart(chart_data.set_index("Epoch"))
            else:
                st.session_state.final_prototypes = None
                st.session_state.prototype_labels = None
                st.session_state.few_shot_strategy = None
                st.session_state.few_shot_trained = False
                st.session_state.model_mode = 'standard'
                st.error("Failed to calculate final prototypes after training.")
                chart_placeholder.empty()
                training_successful = False

            # Rerun only if training finished (successful or not) to clear the form submission state
            # and allow the save section (if successful) to appear correctly outside the form.
            st.rerun()


    # --- END OF TRAINING FORM --- # (Moved Save Section *after* this)

    # --- SAVING SECTION (Displayed only if training was successful in the *last* run) ---
    # This checks the current session state, which reflects the result of the last training attempt.
    if st.session_state.get('final_prototypes') is not None and \
       st.session_state.get('model_mode') == 'few_shot' and \
       st.session_state.get('few_shot_strategy') == 'train_projection':

        st.divider()
        st.subheader("💾 Save Current Few-Shot State")
        current_strategy = st.session_state.get('few_shot_strategy', 'N/A')
        st.info(f"This will save the current model weights (frozen base + trained projection) and prototypes.")

        # Using original keys is fine now as it's outside the form
        save_model_name = st.text_input("Enter a name for this state:", key="save_model_name_input_main")

        # This button is now outside the form, so it's allowed
        if st.button("Save State", key="save_state_button_main"):
            if save_model_name:
                model_to_save = cached_feature_extractor_model()
                model_to_save.eval()
                # Get current class names at the time of saving
                _, _, current_cls_names_for_saving, _ = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)

                save_successful = save_few_shot_state(
                    save_model_name,
                    model_to_save,
                    st.session_state.final_prototypes,
                    st.session_state.prototype_labels,
                    current_cls_names_for_saving,
                    st.session_state.few_shot_strategy
                )

                if save_successful:
                    # Rerun to refresh the saved model list in the sidebar
                    st.rerun()
            else:
                st.warning("Please enter a name before saving.")
    # --- END OF SAVING SECTION ---

# --- END OF `elif option == "Train Few-Shot Model":` ---


# --- Cleanup (Optional - Generally avoid unless necessary) ---
# try:
#     # A better place might be using atexit module if really needed
#     pass # Keep TEMP_DIR for potential caching benefits during session
# except Exception as e:
#     st.sidebar.warning(f"Could not cleanup temp dir: {e}")