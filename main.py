import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import torchvision.transforms as T
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to NuScenes dataset
DATASET_PATH = '/Users/kai/Studium/7.Semester/Projekt_Geodatafusion/v1.0-mini'

# Load NuScenes dataset
nuscenes = NuScenes(version='v1.0-mini', dataroot=DATASET_PATH, verbose=True)


class NuScenesDataset(Dataset):
    def __init__(self, nuscenes, transforms=None):
        self.nuscenes = nuscenes
        self.transforms = transforms
        self.samples = nuscenes.sample  # All samples in the dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get the sample data (front camera image)
        sample_data = self.nuscenes.get('sample_data', sample['data']['CAM_FRONT'])
        image_path = self.nuscenes.get_sample_data_path(sample_data['token'])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get camera intrinsic matrix
        calibrated_sensor = self.nuscenes.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

        # Prepare bounding boxes and labels
        boxes = []
        labels = []
        for ann_token in sample['anns']:
            annotation = self.nuscenes.get('sample_annotation', ann_token)

            # Convert rotation to Quaternion
            rotation = Quaternion(annotation['rotation'])  # Convert to Quaternion

            # Create the Box object with the Quaternion rotation
            box = Box(annotation['translation'], annotation['size'], rotation)

            # Project the 3D bounding box to 2D
            corners_3d = box.corners()
            projected_corners = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]

            # Get min/max coordinates for the 2D box
            x_min, y_min = projected_corners.min(axis=1)
            x_max, y_max = projected_corners.max(axis=1)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # Single class example

        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply any data transformations
        if self.transforms:
            image = self.transforms(image)

        target = {"boxes": boxes, "labels": labels}
        return image, target


# Define a modified view_points function with zero-depth handling
def view_points(points, intrinsic_matrix, normalize=False):
    # Project 3D points using the intrinsic matrix
    points = np.dot(intrinsic_matrix, points)

    # Normalize points if required
    if normalize:
        # Avoid division by zero by setting zero depths to a small epsilon value
        depths = points[2:3, :]
        depths[depths == 0] = 1e-6  # Replace zero depth with epsilon to avoid division by zero
        points = points / depths.repeat(3, 0).reshape(3, points.shape[1])

    return points

# Define dataset transformations
transform = T.Compose([
    T.Resize((800, 800)),  # Resize images to a fixed size
    T.ToTensor(),  # Convert images to tensors
])

# Initialize dataset and dataloader
dataset = NuScenesDataset(nuscenes, transforms=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')  # Update 'pretrained' to 'weights'

# Modify the final layer to match the number of classes (background + NuScenes categories)
num_classes = 2  # 1 class + background (for this example, assume a single class)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move model to device
model = model.to(device)

# Define optimizer and training parameters
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 5

# Training loop
model.train()  # Set the model to training mode
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Check for NaN losses
        if torch.isnan(losses):
            print("Loss is NaN, stopping training.")
            break

        # Backward pass and optimizer step
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)}")

# Set the model to evaluation mode
model.eval()

# Load a test image and visualize the predictions
test_image, _ = dataset[0]  # Load an image from the dataset

with torch.no_grad():
    prediction = model([test_image.to(device)])

# Visualize results
plt.imshow(test_image.permute(1, 2, 0).cpu().numpy())
for box in prediction[0]['boxes']:
    plt.plot([box[0], box[2]], [box[1], box[3]], 'r-')
plt.show()

# Export the model to ONNX for MATLAB compatibility
import torch.onnx

# Dummy input for exporting model to ONNX
dummy_input = torch.randn(1, 3, 800, 800).to(device)
torch.onnx.export(model, dummy_input, "faster_rcnn_nuscenes.onnx", opset_version=11)
print("Model exported to ONNX format.")