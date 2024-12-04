import os
from sklearn.model_selection import train_test_split # type: ignore
import shutil

dataset_path = "datasets/v2-data"
labels = ["Bengin", "Malignant", "Normal"]

data_folder = os.path.join(dataset_path, 'Data')

# Create train and test directories
train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for label in labels:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)

# Split data
for label in labels:
    label_folder = os.path.join(data_folder, label)
    images = os.listdir(label_folder)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
    for image in train_images:
        shutil.copy(os.path.join(label_folder, image), os.path.join(train_dir, label, image))
    
    for image in val_images:
        shutil.copy(os.path.join(label_folder, image), os.path.join(val_dir, label, image))