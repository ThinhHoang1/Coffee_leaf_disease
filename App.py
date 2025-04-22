# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import os
import tempfile
import random
import shutil
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import pandas as pd
st.set_page_config(layout="wide")
from PIL import Image, ImageOps
import streamlit as st
import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create the users table (if it doesn't exist)
def create_table():
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT,
        name TEXT,
        password TEXT
    )''')
    conn.commit()

# Register a new user
def register_user(user_id, user_name, password):
    c.execute('INSERT INTO users (user_id, username, name, password) VALUES (?, ?, ?, ?)', (user_id, user_name, user_name, password))
    conn.commit()

# Check if the user is registered (login validation)
def check_login(user_id, password):
    c.execute('SELECT * FROM users WHERE user_id = ? AND password = ?', (user_id, password))
    data = c.fetchone()
    return data

# --- Session Management ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_name = None

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Using device: {DEVICE}")

# --- IMPORTANT: Paths ---
# Using relative paths or environment variables is recommended for portability
# Replace these with your actual paths or use a configuration method
BASE_DATASET = "App data/data_optimize/Basee_data"  # Example: Use forward slashes or raw strings r"E:\..."
RARE_DATASET = "App data/data_optimize/Rare data/"  # Example: Use forward slashes or raw strings r"E:\..."

# Check if paths exist
if not os.path.isdir(BASE_DATASET):
    st.error(f"Base dataset directory not found: {BASE_DATASET}")
    st.stop()
# Create rare dataset directory if it doesn't exist
os.makedirs(RARE_DATASET, exist_ok=True)
# if not os.path.isdir(RARE_DATASET):
#     st.error(f"Rare dataset directory not found: {RARE_DATASET}")
#     st.stop()

# Path to the saved model weights - Ensure this exists
MODEL_WEIGHTS_PATH = "model/efficientnet_coffee (1).pth" # Example: Use forward slashes
if not os.path.isfile(MODEL_WEIGHTS_PATH):
    st.error(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")
    st.stop()

TEMP_DIR = tempfile.mkdtemp() # This is fine

# EfficientNet feature extractor with projection layer
class EfficientNetWithProjection(nn.Module):
    def __init__(self, base_model, output_dim=1024):
        super(EfficientNetWithProjection, self).__init__()
        self.model = base_model # This holds the EfficientNet base (feature extractor part)
        # Determine the input feature size dynamically from the base model if possible
        # For EfficientNetB0, the layer before the classifier has 1280 features
        in_features = 1280
        # Example of trying to get it dynamically (might fail depending on model structure)
        # try:
        #     # This assumes the base_model has a structure where the features can be accessed
        #     # Adjust based on actual structure after removing classifier
        #     # For EfficientNet, the features are often pooled before the classifier.
        #     # Let's assume the base_model output IS the feature vector (e.g., after Identity())
        #     # We might need to run a dummy input to know the size, or hardcode it based on known arch.
        #     # Hardcoding is safer if the arch is fixed (like EfficientNetB0 -> 1280)
        #      pass # Keep hardcoded 1280 for now
        # except AttributeError:
        #      st.warning("Could not dynamically determine input features for projection layer. Using default 1280.")
        #      in_features = 1280 # Default for EfficientNetB0

        self.projection = nn.Linear(in_features, output_dim) # Projection layer

    def forward(self, x):
        # Pass input through the base model (feature extractor part)
        # Decide if base_model should be frozen here or during training setup
        # If frozen here, it's *always* frozen. If during training, it's configurable.
        # Let's keep it trainable by default here, and freeze during training setup if requested.
        features = self.model(x) # Get features from EfficientNet base
        return self.projection(features) # Project to output_dim dimensions

# --- Model Loading Functions ---

# Base EfficientNet model structure (used for loading weights)
def get_base_efficientnet_architecture(num_classes=5):
    model = models.efficientnet_b0(weights=None) # Load architecture only
    in_features = model.classifier[1].in_features # Get feature dimension
    model.classifier[1] = nn.Linear(in_features, num_classes) # Adjust final layer
    return model

# Feature Extractor model structure (for few-shot)
# This function now prepares the base model first, then wraps it.
def get_feature_extractor_base():
    # Load EfficientNet pre-trained on ImageNet (provides good initial features)
    base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # We need to load our coffee-tuned weights into this structure
    # To do this safely, first modify it to match the saved state dict (5 classes)
    in_features = base_model.classifier[1].in_features
    base_model.classifier[1] = nn.Linear(in_features, 5) # Match saved model's output layer

    # Load the coffee-specific weights
    try:
        # Use weights_only=True for security
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        # Load weights, allowing for missing keys if projection wasn't part of saved model
        missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
        if missing_keys: st.sidebar.warning(f"Weights load: Missing keys: {missing_keys}")
        if unexpected_keys: st.sidebar.warning(f"Weights load: Unexpected keys: {unexpected_keys}")
        # st.sidebar.info("Loaded custom coffee weights into base model.") # Less verbose
    except Exception as e:
        st.error(f"Error loading model weights from {MODEL_WEIGHTS_PATH}: {e}")
        st.stop()

    # Remove the classifier to use it as a feature extractor
    base_model.classifier = nn.Identity()
    base_model.eval() # Set base model to eval mode
    return base_model

# Function to load the standard classifier model (for reset/fallback)
def load_standard_classifier():
    model = get_base_efficientnet_architecture(num_classes=5)
    try:
        # Use weights_only=True for security
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=True) # Use strict=True for the final classifier
        # st.sidebar.info("Loaded standard classifier weights.") # Less verbose
    except Exception as e:
        st.error(f"Error loading model weights for standard classifier: {e}")
        st.stop()
    model.to(DEVICE)
    model.eval()
    return model


# --- Caching ---
# Cache the feature extractor model resource (Base + Projection)
@st.cache_resource
def cached_feature_extractor_model():
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
        full_dataset = datasets.ImageFolder(base_path, transform)
        num_base_classes = len(full_dataset.classes)
        # st.sidebar.write(f"Base classes found: {num_base_classes}") # Less verbose

        # Handle potential empty rare dataset
        rare_classes_found = 0
        if os.path.exists(rare_path) and os.listdir(rare_path):
             try:
                rare_dataset = datasets.ImageFolder(rare_path, transform)
                if len(rare_dataset.samples) > 0:
                    # Important: Adjust labels for rare classes
                    rare_dataset.samples = [(path, label + num_base_classes) for path, label in rare_dataset.samples]
                    combined_dataset = ConcatDataset([full_dataset, rare_dataset])
                    rare_classes_found = len(rare_dataset.classes)
                    # st.sidebar.write(f"Rare classes found: {rare_classes_found}") # Less verbose
                else:
                    combined_dataset = full_dataset
                    # st.sidebar.write("Rare dataset folder exists but contains no images.") # Less verbose
             except Exception as e_rare:
                  st.warning(f"Could not load rare dataset from {rare_path}: {e_rare}. Using base dataset only.")
                  combined_dataset = full_dataset
        else:
             combined_dataset = full_dataset
             # st.sidebar.write("No rare classes found or directory empty.") # Less verbose

        # Create class indices mapping
        indices = {}
        all_samples = []
        if isinstance(combined_dataset, ConcatDataset):
            # Correctly iterate through ConcatDataset samples
            cumulative_len = 0
            for i, ds in enumerate(combined_dataset.datasets):
                 for path, label in ds.samples:
                      # The label should already be correct (offset applied above)
                      img_idx_in_concat = cumulative_len + ds.samples.index((path, label)) # Find index within sub-dataset
                      # This indexing might be fragile if samples aren't unique, better iterate with enumerate
                      pass # Let's use enumerate on the combined list instead

            # Rebuild all_samples correctly ensuring index corresponds to ConcatDataset index
            concat_idx = 0
            for ds in combined_dataset.datasets:
                for path, label in ds.samples:
                     all_samples.append((path, label)) # Label already adjusted
                     indices.setdefault(label, []).append(concat_idx)
                     concat_idx += 1

        else: # It's just the base dataset
            for idx, (path, label) in enumerate(combined_dataset.samples):
                 all_samples.append((path, label))
                 indices.setdefault(label, []).append(idx)


        # Get class names in the correct order (base first, then rare sorted)
        base_classes = sorted(full_dataset.classes)
        rare_classes = []
        if rare_classes_found > 0 and 'rare_dataset' in locals():
            rare_classes = sorted(rare_dataset.classes)
        class_names = base_classes + rare_classes

        st.sidebar.metric("Base Classes", num_base_classes)
        st.sidebar.metric("Rare Classes", rare_classes_found)
        st.sidebar.metric("Total Classes", len(class_names))

        if len(class_names) == 0:
             st.error("No classes found in base or rare datasets. Please check dataset paths and contents.")
             st.stop()

        return combined_dataset, indices, class_names, num_base_classes

    except FileNotFoundError as e:
        st.error(f"Dataset path error: {e}. Please check BASE_DATASET and RARE_DATASET paths.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        st.exception(e) # Show full traceback
        st.stop()

# Image transform
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
        # st.warning(f"Not enough classes ({len(available_classes)}) available to form an N-way={n_way} episode. Trying with {len(available_classes)} classes.")
        n_way = len(available_classes) # Adjust n_way if not enough classes
        if n_way < 2: # Need at least 2 classes for an episode
             st.error("Cannot create an episode, less than 2 classes available in total.")
             return None, None, None, None

    selected_class_ids = random.sample(available_classes, n_way) # Sample based on adjusted n_way
    support_imgs, query_imgs = [], []
    support_labels, query_labels = [], []
    # Map original label to 0..n_way-1 for THIS episode
    episode_class_map = {class_id: i for i, class_id in enumerate(selected_class_ids)}
    actual_n_way = 0 # Track how many classes actually get included

    for original_label in selected_class_ids:
        indices = class_indices.get(original_label, []) # Use .get for safety

        # Check if enough samples exist for the class
        min_samples_needed = n_shot + n_query
        if len(indices) < min_samples_needed:
            # Only show warning once per run maybe? Suppressing for now.
            # st.warning(f"Skipping class '{class_list[original_label]}' (ID: {original_label}) in this episode: Not enough samples ({len(indices)} found, {min_samples_needed} needed).")
            continue # Skip this class for this episode

        sampled_indices = random.sample(indices, min_samples_needed)

        # Get images from the dataset using the sampled indices
        try:
            support_imgs += [dataset[i][0] for i in sampled_indices[:n_shot]]
            query_imgs += [dataset[i][0] for i in sampled_indices[n_shot:]]
        except IndexError as e:
             # This error implies indices mapping might be wrong if it occurs
             st.error(f"IndexError during episode creation: {e}. Index: {i}, Dataset size: {len(dataset)}. Check dataset indexing/class_indices.")
             return None, None, None, None
        except Exception as e:
             st.error(f"Error retrieving data during episode creation: {e}")
             return None, None, None, None

        # Assign new sequential labels (0 to n_way-1) for the episode
        new_label = episode_class_map[original_label]
        support_labels += [new_label] * n_shot
        query_labels += [new_label] * n_query
        actual_n_way += 1 # Increment count of classes successfully added

    if not support_imgs or not query_imgs or actual_n_way < 2: # Check if enough classes were added
         # st.warning(f"Episode creation resulted in < 2 valid classes ({actual_n_way}). Skipping.") # Reduce verbosity
         return None, None, None, None

    # Return tensors on the correct device
    try:
        return (torch.stack(support_imgs).to(DEVICE),
                torch.tensor(support_labels, dtype=torch.long).to(DEVICE),
                torch.stack(query_imgs).to(DEVICE),
                torch.tensor(query_labels, dtype=torch.long).to(DEVICE))
    except Exception as e:
        st.error(f"Error stacking tensors in create_episode: {e}")
        return None, None, None, None


# Loss function for Few-Shot Learning (Prototypical Networks)
def proto_loss(support_embeddings, support_labels, query_embeddings, query_labels):
    """Calculates the Prototypical Network loss and accuracy."""
    if support_embeddings is None or support_embeddings.size(0) == 0 or \
       query_embeddings is None or query_embeddings.size(0) == 0:
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0

    unique_labels = torch.unique(support_labels)
    n_way_actual = len(unique_labels)

    if n_way_actual < 2: # Need at least 2 classes for meaningful loss/accuracy
        return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0

    prototypes = []
    query_indices_for_loss = []
    mapped_query_labels_for_loss = []
    label_map = {} # Map original episode label to 0..n_way_actual-1 index

    for i, label in enumerate(unique_labels):
        label_map[label.item()] = i
        class_embeddings = support_embeddings[support_labels == label]
        if class_embeddings.size(0) > 0:
             prototypes.append(class_embeddings.mean(dim=0))
        else:
             # Should not happen if create_episode checks correctly
             st.warning(f"No support embeddings found for label {label} during loss calculation.")
             # Handle this? Maybe return zero loss or skip class? Returning zero loss for safety.
             return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0

    if len(prototypes) < n_way_actual: # Check if all classes yielded a prototype
         st.warning("Mismatch between unique labels and calculated prototypes.")
         return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0

    prototypes = torch.stack(prototypes)

    # Filter query embeddings/labels to only include those whose classes are in the support set prototypes
    # This is crucial if create_episode sometimes yields partial episodes
    valid_query_mask = torch.isin(query_labels, unique_labels)
    if not torch.any(valid_query_mask):
         return torch.tensor(0.0, requires_grad=True).to(DEVICE), 0.0 # No valid query samples

    filtered_query_embeddings = query_embeddings[valid_query_mask]
    filtered_query_labels = query_labels[valid_query_mask]

    distances = torch.cdist(filtered_query_embeddings, prototypes)
    predictions = torch.argmin(distances, dim=1)

    # Map the filtered original query labels to the 0..n_way_actual-1 range
    mapped_query_labels = torch.tensor([label_map[lbl.item()] for lbl in filtered_query_labels], dtype=torch.long).to(DEVICE)

    correct_predictions = (predictions == mapped_query_labels).sum().item()
    total_predictions = mapped_query_labels.size(0)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    loss = F.cross_entropy(-distances, mapped_query_labels)

    return loss, accuracy


# Recalculate prototypes after training for the entire dataset
@st.cache_data # Cache based on model state? Might need manual clearing. Let's clear manually before calculation.
def calculate_final_prototypes(_model, _dataset, _class_names):
    st.info("Calculating final prototypes for all classes...")
    _model.eval()
    all_embeddings = {}

    # Use a DataLoader
    loader = DataLoader(_dataset, batch_size=64, shuffle=False, num_workers=0) # Increased batch size

    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            emb = _model(imgs)
            for i in range(emb.size(0)):
                label = labs[i].item()
                if label not in all_embeddings:
                    all_embeddings[label] = []
                all_embeddings[label].append(emb[i].cpu().clone()) # Use clone to be safe

    final_prototypes = []
    prototype_labels = []
    unique_labels_present = sorted(list(all_embeddings.keys()))

    if not unique_labels_present:
        st.warning("No embeddings were generated. Cannot calculate prototypes.")
        return None, None

    for label in unique_labels_present:
        if label < 0 or label >= len(_class_names):
             st.warning(f"Skipping label {label} during prototype calculation: Out of bounds for class names (len={len(_class_names)}).")
             continue

        class_embeddings_list = all_embeddings[label]
        if class_embeddings_list:
            class_embeddings = torch.stack(class_embeddings_list)
            prototype = class_embeddings.mean(dim=0)
            final_prototypes.append(prototype)
            prototype_labels.append(label)
        # else: # This case shouldn't happen if label is in unique_labels_present
            # st.warning(f"No embeddings found for class ID {label} ('{_class_names[label]}') during final prototype calculation.")


    if not final_prototypes:
         st.warning("Could not calculate any final prototypes.")
         return None, None

    st.success(f"Calculated {len(final_prototypes)} final prototypes.")
    return torch.stack(final_prototypes).to(DEVICE), prototype_labels


# Visualize Prototypes Function
def visualize_prototypes(prototypes_tensor, prototype_labels, class_names_list):
    st.write("Visualizing Prototypes using PCA...")

    if prototypes_tensor is None or prototypes_tensor.size(0) == 0:
        st.warning("⚠️ No prototypes available to visualize.")
        return

    if len(prototypes_tensor) < 2:
        st.warning("⚠️ Need at least 2 prototypes to visualize with PCA.")
        return

    pca = PCA(n_components=2)
    try:
        prototypes_np = prototypes_tensor.cpu().detach().numpy()
        prototypes_2d = pca.fit_transform(prototypes_np)

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, label_index in enumerate(prototype_labels):
            if 0 <= label_index < len(class_names_list):
                class_name = class_names_list[label_index]
                ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1], label=f"{class_name} (ID: {label_index})", s=100)
            else:
                ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1], label=f"Unknown Label (ID: {label_index})", s=100, marker='x', color='red')
                st.warning(f"Label index {label_index} out of bounds for class names list (length {len(class_names_list)}).")

        ax.set_title("Prototypes Visualization (PCA)")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend(loc='best')
        ax.grid(True)
        st.pyplot(fig)

    except ValueError as ve:
         st.error(f"PCA Error: {ve}. Ensure prototypes are valid numbers.")
    except Exception as e:
        st.error(f"Error during PCA visualization: {e}")
        st.exception(e)




# YoLo
# Load YOLOv11 model (use the trained model from training step)
@st.cache_resource
def load_yolo_model():
    model_weights_path ="model/best.pt"  # Path to the trained model
    model = YOLO(model_weights_path)  # Load the trained YOLOv11 model
    model.to(DEVICE)  # Move to GPU if available
    model.eval()  # Set to evaluation mode
    return model

# Detection function for YOLOv11
def detect_objects(image):
    model = load_yolo_model()  # Load the YOLOv11 model
    
    # Convert image to format accepted by YOLO model
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for YOLO
    
    # Perform inference
    results = model(img_array)  # Run the model on the image
    
    # Render results (this will return a list of images with bounding boxes)
    result_image = results[0].plot()  # Simply plot without extra arguments
    
    # Convert back to RGB for display in Streamlit
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Access model class names
    class_names = model.names  # Get the class names from the model
    
    # Extract detection details manually
    detection_details = {
        "xmin": results[0].boxes.xyxy[:, 0].cpu().numpy(),  # x_min
        "ymin": results[0].boxes.xyxy[:, 1].cpu().numpy(),  # y_min
        "xmax": results[0].boxes.xyxy[:, 2].cpu().numpy(),  # x_max
        "ymax": results[0].boxes.xyxy[:, 3].cpu().numpy(),  # y_max
        "confidence": results[0].boxes.conf.cpu().numpy(),  # Confidence score
        "class": [class_names[i] for i in results[0].boxes.cls.cpu().numpy().astype(int)],  # Class names
    }
    
    # Convert detection details to a DataFrame
    detections = pd.DataFrame(detection_details)
    
    return result_image, detections

# Call to ensure the users table is created when app is first run
create_table()

# --- Session Management ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
# === Main App Logic ===
st.title("🌿 Coffee Leaf Disease Classifier + Few-Shot Learning + Detection")
# --- User ID Input ---
# --- Login Section ---
# --- User ID Input ---
# --- Login Section ---
if not st.session_state.logged_in:
    menu = ["Login", "Register"]
    choice = st.sidebar.radio("Select an action", menu)
    
    if choice == "Login":
        st.header("🔐 Login to the App")
        
        # Login form
        with st.form("login_form"):
            user_id = st.text_input("Enter your User ID:")
            password = st.text_input("Enter your Password:", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                user_data = check_login(user_id, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.user_name = user_data[2]  # assuming the 'name' column stores the user's name
                    st.success(f"Welcome, {st.session_state.user_name}!")
                else:
                    st.error("Invalid User ID or Password")
    
    elif choice == "Register":
        st.subheader("New to the app? Register here:")

        # Registration form
        with st.form("register_form"):
            new_user_id = st.text_input("Choose a User ID:")
            new_name = st.text_input("Your Name:")
            new_password = st.text_input("Choose a Password:", type="password")
            register_button = st.form_submit_button("Register")
            
            if register_button:
                if new_user_id and new_name and new_password:
                    # Check if the user ID already exists
                    c.execute('SELECT * FROM users WHERE user_id = ?', (new_user_id,))
                    existing_user = c.fetchone()
                    if existing_user:
                        st.error("User ID already exists. Please choose a different one.")
                    else:
                        register_user(new_user_id, new_name, new_password)
                        st.success(f"New user {new_name} created! Please log in.")
                else:
                    st.error("Please fill in all the fields.")

else:
    # After login, show the rest of the app (disease classifier, few-shot learning, etc.)
    st.header(f"👋 Hello, {st.session_state.user_name}!")
    
    # Show options after logging in
    option = st.radio(
        "Choose an action:",
        ["Upload & Predict", "Add/Manage Rare Classes", "Train Few-Shot Model", "Visualize Prototypes", "Detection"],
        horizontal=True, key="main_option"
    )
# --- Initialize Session State ---
if 'few_shot_trained' not in st.session_state:
    st.session_state.few_shot_trained = False
if 'final_prototypes' not in st.session_state:
    st.session_state.final_prototypes = None # Tensor of prototypes
if 'prototype_labels' not in st.session_state:
    st.session_state.prototype_labels = None # List of labels corresponding to prototypes
if 'model_mode' not in st.session_state:
     # Default to standard unless few-shot has been successfully trained before
     st.session_state.model_mode = 'standard'

# --- Load Data ---
# This runs once and caches results, or re-runs if cache is cleared
combined_dataset, class_indices, class_names, num_base_classes = get_combined_dataset_and_indices(BASE_DATASET, RARE_DATASET)

# --- Sidebar and Mode Selection ---
st.sidebar.header("Options")

# Button to switch back to standard classification
if st.sidebar.button("🔄 Use Standard Classifier"):
    st.session_state.model_mode = 'standard'
    st.session_state.few_shot_trained = False # Reset trained flag if switching manually
    st.success("Switched to Standard Classification Mode.")
    st.rerun()

# Display current mode
current_mode_display = "Standard Classifier"
if st.session_state.model_mode == 'few_shot' and st.session_state.final_prototypes is not None:
     current_mode_display = "Few-Shot (Prototypes)"
st.sidebar.write(f"**Current Mode:** {current_mode_display}")
if st.session_state.model_mode == 'few_shot' and st.session_state.final_prototypes is not None:
     st.sidebar.write(f"*{len(st.session_state.final_prototypes)} prototypes active.*")


# --- Main Panel Options ---
if st.session_state.user_id:
    # PLACE YOUR MAIN LOGIC BELOW THIS CHECK
    # For example: the st.radio option section and everything that follows
    option = st.radio(
        "Choose an action:",
        ["Upload & Predict", "Add/Manage Rare Classes", "Train Few-Shot Model", "Visualize Prototypes", "Detection"],
        horizontal=True, key="main_option"
    )

    # (keep rest of your code unchanged here)

else:
    st.stop()  # Prevent rest of the app from running


# Select the appropriate model based on mode
# We load both and use the one needed, caching helps
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
                            st.session_state.prototype_labels is not None)

            if use_few_shot:
                # --- Few-Shot Prototype Prediction ---
                st.subheader("Prediction using Prototypes")
                model_to_use = feature_extractor_model
                model_to_use.eval()
                with torch.no_grad():
                    embedding = model_to_use(input_tensor)
                    distances = torch.cdist(embedding, st.session_state.final_prototypes)
                    pred_prototype_index = torch.argmin(distances, dim=1).item()
                    predicted_label = st.session_state.prototype_labels[pred_prototype_index]

                    if 0 <= predicted_label < len(class_names):
                         predicted_class_name = class_names[predicted_label]
                         confidence_scores = torch.softmax(-distances, dim=1)
                         confidence = confidence_scores[0, pred_prototype_index].item()
                         st.metric(label="Prediction (Prototype)", value=predicted_class_name, delta=f"{confidence * 100:.1f}% Confidence")
                    else:
                         st.error(f"Predicted label index {predicted_label} is out of range for known class names.")

            else:
                # --- Standard Classification Prediction ---
                st.subheader("Prediction using Standard Classifier")
                if st.session_state.model_mode != 'standard':
                     st.warning("Falling back to Standard Classifier mode for this prediction.")
                model_to_use = standard_classifier_model
                model_to_use.eval()
                with torch.no_grad():
                    outputs = model_to_use(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_label = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_label].item()

                    if 0 <= pred_label < num_base_classes: # Check against NUM_BASE_CLASSES
                        predicted_class_name = class_names[pred_label]
                        st.metric(label="Prediction (Standard)", value=predicted_class_name, delta=f"{confidence * 100:.1f}% Confidence")
                    else:
                        st.error(f"Standard classifier predicted label {pred_label}, which is out of range for base classes ({num_base_classes}).")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)
elif option == "Detection":
    st.header("🕵️ Object Detection with YOLOv11")
    uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        result_image, detections = detect_objects(image)

        # Resize for display (maintain aspect ratio)
        display_image = Image.fromarray(result_image)
        display_image = ImageOps.contain(display_image, (800, 600))  # Resize while keeping aspect ratio

        st.image(display_image, caption="Detection Result")

        # Show detection results
        if not detections.empty:
            st.subheader("📋 Detection Results:")
            st.dataframe(detections)
        else:
            st.info("No objects detected.")

elif option == "Add/Manage Rare Classes":
    st.header("➕ Add New Rare Class (Few-Shot)")
    st.write(f"Upload exactly 5 sample images for the new disease class. They will be saved in: `{RARE_DATASET}`")

    with st.form("add_class_form"):
        new_class_name = st.text_input("Enter the name for the new rare class:")
        uploaded_files = st.file_uploader(
            "Upload 5 images:", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key="add_class_uploader"
        )
        submitted = st.form_submit_button("Add Class")

        if submitted:
            if not new_class_name:
                st.warning("Please enter a class name.")
            elif len(uploaded_files) != 5:
                st.warning(f"Please upload exactly 5 images. You uploaded {len(uploaded_files)}.")
            else:
                sanitized_class_name = "".join(c for c in new_class_name if c.isalnum() or c in (' ', '_')).rstrip().replace(" ", "_")
                new_class_dir = os.path.join(RARE_DATASET, st.session_state.user_id, sanitized_class_name)


                if os.path.exists(new_class_dir):
                    st.warning(f"A class directory named '{sanitized_class_name}' already exists. Choose a different name or delete the existing one first.")
                else:
                    try:
                        os.makedirs(new_class_dir, exist_ok=True)
                        image_save_errors = 0
                        for i, file in enumerate(uploaded_files):
                            try:
                                img = Image.open(file).convert("RGB")
                                save_path = os.path.join(new_class_dir, f"sample_{i+1}.jpg")
                                img.save(save_path)
                                # st.write(f"Saved: {save_path}") # Reduce verbosity
                            except Exception as img_e:
                                st.error(f"Error saving image {i+1} ({file.name}): {img_e}")
                                image_save_errors += 1
                        if image_save_errors == 0:
                             st.success(f"✅ Added new class: '{sanitized_class_name}'. Please re-run 'Train Few-Shot Model' to incorporate it.")
                             st.cache_data.clear() # Clear dataset cache
                             # Clear potentially outdated prototypes if a class is added
                             st.session_state.final_prototypes = None
                             st.session_state.prototype_labels = None
                             st.session_state.few_shot_trained = False
                             st.rerun()
                        else:
                             st.error(f"Failed to save {image_save_errors} images. Class may be incomplete.")

                    except Exception as e:
                        st.error(f"Error creating directory or saving images for class '{sanitized_class_name}': {e}")

    st.divider()
    st.header("❌ Delete a Rare Class")

    try:
        user_rare_dir = os.path.join(RARE_DATASET, st.session_state.user_id)
        rare_class_dirs = [d for d in os.listdir(user_rare_dir) if os.path.isdir(os.path.join(user_rare_dir, d))]
        
        if not rare_class_dirs:
            st.info("No rare classes found to delete.")
        else:
            with st.form("delete_class_form"):
                to_delete = st.selectbox("Select rare class to delete:", rare_class_dirs)
                confirm_delete = st.checkbox(f"Are you sure you want to delete '{to_delete}' and its contents?")
                delete_submit = st.form_submit_button("Delete Class")

                if delete_submit:
                    if confirm_delete:
                        delete_path = os.path.join(RARE_DATASET, to_delete)
                        try:
                            shutil.rmtree(delete_path)
                            st.success(f"✅ Deleted rare class: {delete_path}")
                            st.cache_data.clear()
                            st.session_state.few_shot_trained = False
                            st.session_state.final_prototypes = None
                            st.session_state.prototype_labels = None
                            st.session_state.model_mode = 'standard'
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting directory {delete_path}: {e}")
                    else:
                        st.warning("Please confirm the deletion by checking the box.")

    except FileNotFoundError:
        st.info(f"Rare dataset directory '{RARE_DATASET}' not found.")
    except Exception as e:
        st.error(f"Error listing rare classes: {e}")



elif option == "Train Few-Shot Model":
    st.header("🚀 Train Few-Shot Model")
    st.warning("⚠️ This will fine-tune the feature extractor. Performance on original classes might change.")

    if len(class_names) <= 1:
         st.error("Need at least two classes in the combined dataset to perform few-shot training.")
    else:
        st.info(f"Training will use all {len(class_names)} available classes (Base + Rare).")

        with st.form("train_form"):
            st.write("Configure Training Parameters:")
            cols = st.columns(3)
            with cols[0]:
                epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5, step=1)
                n_way_train = st.number_input("N-way", min_value=2, max_value=min(len(class_names), 10), value=min(len(class_names), 5), step=1, help="Classes per Episode")
            with cols[1]:
                episodes_per_epoch = st.number_input("Episodes/Epoch", min_value=1, max_value=500, value=50, step=10) # Increased default
                n_shot = st.number_input("N-shot", min_value=1, max_value=10, value=5, step=1, help="Support Images/Class")
            with cols[2]:
                learning_rate = st.number_input("Learning Rate", min_value=1e-7, max_value=1e-3, value=1e-5, step=1e-6, format="%e")
                n_query = st.number_input("N-query", min_value=1, max_value=10, value=5, step=1, help="Query Images/Class")

            freeze_backbone = st.checkbox("❄️ Freeze Backbone Layers (Train only projection layer)", value=True, help="Recommended to reduce forgetting.")

            submitted = st.form_submit_button("Start Training")

            if submitted:
                st.info("Starting few-shot training...")
                # Ensure the feature extractor model is loaded for training
                model = cached_feature_extractor_model()
                model.train() # Set model to training mode (important!)

                # --- Optimizer Setup ---
                trainable_params = []
                if freeze_backbone:
                    st.info("Freezing base EfficientNet backbone...")
                    try:
                        # Freeze base model parameters
                        for param in model.model.parameters():
                             param.requires_grad = False
                        # Ensure projection layer parameters are trainable
                        for param in model.projection.parameters():
                             param.requires_grad = True
                        # Collect only trainable parameters (should be just projection layer)
                        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                        if not trainable_params:
                             st.error("No trainable parameters found after attempting to freeze backbone! Check model structure.")
                             st.stop()
                        st.success("Base backbone frozen. Training projection layer only.")
                    except AttributeError:
                         st.error("Could not access model.model or model.projection to freeze/unfreeze parameters. Training all layers.")
                         trainable_params = list(model.parameters()) # Fallback
                else:
                    st.info("Training all layers (backbone + projection).")
                    # Ensure all params are trainable if not freezing
                    for param in model.parameters():
                        param.requires_grad = True
                    trainable_params = list(model.parameters())


                optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-5)
                st.info(f"Using Adam optimizer with LR={learning_rate:.0e}")

                # Training Loop
                progress_bar = st.progress(0)
                loss_history = []
                accuracy_history = []
                status_placeholder = st.empty() # Placeholder for status text

                total_steps = epochs * episodes_per_epoch
                current_step = 0

                for epoch in range(epochs):
                    epoch_loss = 0.0
                    epoch_accuracy = 0.0
                    valid_episodes_in_epoch = 0

                    for episode in range(episodes_per_epoch):
                        current_step += 1
                        # Create episode
                        s_imgs, s_labels, q_imgs, q_labels = create_episode(
                            combined_dataset, class_indices, class_names, n_way=n_way_train, n_shot=n_shot, n_query=n_query
                        )

                        if s_imgs is None: continue # Skip if episode creation failed

                        optimizer.zero_grad()
                        try:
                            s_emb = model(s_imgs)
                            q_emb = model(q_imgs)
                        except Exception as model_e:
                             st.error(f"Error during model forward pass in training: {model_e}")
                             continue # Skip episode on model error

                        loss, accuracy = proto_loss(s_emb, s_labels, q_emb, q_labels)

                        if loss is not None and not torch.isnan(loss) and loss.requires_grad:
                             try:
                                loss.backward()
                                optimizer.step()
                                epoch_loss += loss.item()
                                epoch_accuracy += accuracy
                                valid_episodes_in_epoch += 1
                             except Exception as optim_e:
                                  st.error(f"Error during optimizer step or backward pass: {optim_e}")
                                  # Consider stopping or just skipping step? Skipping for now.
                        # else: # Reduce verbosity
                            # st.warning(f"Skipping episode {episode+1}/{episodes_per_epoch} due to invalid loss.")

                        # Update status and progress bar less frequently to avoid slowing down
                        if (episode + 1) % 10 == 0 or episode == episodes_per_epoch - 1:
                              progress = current_step / total_steps
                              progress_bar.progress(progress)
                              status_placeholder.text(f"Epoch {epoch+1}/{epochs} | Episode {episode+1}/{episodes_per_epoch} | Progress: {progress*100:.1f}%")


                    # Log epoch results
                    if valid_episodes_in_epoch > 0:
                        avg_loss = epoch_loss / valid_episodes_in_epoch
                        avg_accuracy = epoch_accuracy / valid_episodes_in_epoch
                        loss_history.append(avg_loss)
                        accuracy_history.append(avg_accuracy)
                        # Update status text at end of epoch
                        status_placeholder.text(f"Epoch {epoch+1}/{epochs} Completed - Avg Loss: {avg_loss:.4f} - Avg Accuracy: {avg_accuracy*100:.2f}%")
                    else:
                         status_placeholder.text(f"Epoch {epoch+1}/{epochs} Completed - No valid episodes were run.")


                status_placeholder.success("✅ Few-Shot Training Finished!")

                # --- Final Prototype Calculation ---
                st.info("Calculating final prototypes based on the trained model...")
                st.cache_data.clear() # Clear data cache before recalculating
                final_prototypes_tensor, final_prototype_labels = calculate_final_prototypes(model, combined_dataset, class_names)

                st.session_state.final_prototypes = final_prototypes_tensor
                st.session_state.prototype_labels = final_prototype_labels
                st.session_state.few_shot_trained = True
                st.session_state.model_mode = 'few_shot' # Switch to few-shot mode

                if final_prototypes_tensor is not None:
                    st.success(f"Prototypes updated for {len(final_prototype_labels)} classes. Model is now in Few-Shot Prediction mode.")
                    # Display training curves
                    chart_data = {"Epoch": list(range(1, epochs + 1)), "Loss": loss_history, "Accuracy": accuracy_history}
                    st.line_chart(chart_data, x="Epoch", y=["Loss", "Accuracy"])
                else:
                    st.error("Failed to calculate final prototypes after training.")


elif option == "Visualize Prototypes":
    st.header("📊 Visualize Class Prototypes")
    if st.session_state.final_prototypes is not None and st.session_state.prototype_labels is not None:
        visualize_prototypes(st.session_state.final_prototypes, st.session_state.prototype_labels, class_names)
    else:
        st.warning("⚠️ No prototypes calculated yet. Run 'Train Few-Shot Model' or calculate them below.")

    st.divider()
    st.info("You can calculate prototypes based on the current feature extractor state (useful before/after training).")
    if st.button("Calculate/Recalculate Prototypes Now"):
         with st.spinner("Calculating..."):
              proto_model = cached_feature_extractor_model() # Always use feature extractor for prototypes
              proto_model.eval()
              st.cache_data.clear() # Clear cache before calculation
              temp_prototypes, temp_labels = calculate_final_prototypes(proto_model, combined_dataset, class_names)
              if temp_prototypes is not None:
                   st.session_state.final_prototypes = temp_prototypes
                   st.session_state.prototype_labels = temp_labels
                   # Don't set few_shot_trained=True here, as it wasn't explicit training via the form
                   st.success("Prototypes calculated/recalculated.")
                   st.rerun() # Rerun to show visualization
              else:
                   st.error("Failed to calculate prototypes.")

# --- Footer/Cleanup ---
# Consider cleanup of TEMP_DIR if necessary, e.g. using atexit module
# import atexit
# def cleanup():
#     print(f"Cleaning up temporary directory: {TEMP_DIR}")
#     shutil.rmtree(TEMP_DIR, ignore_errors=True)
# atexit.register(cleanup)