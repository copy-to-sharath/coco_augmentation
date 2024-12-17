import os
import json
import random
import shutil
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from matplotlib import pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# Directory structure
# .
# |-- coco_dataset
# |     |-- annotations
# |     |     |-- instances_train.json
# |     |-- images
# |           |-- train
# |           |-- augmented
# |-- augmented_dataset
# |-- augment_coco.py

# Set paths for dataset
coco_path = '/media/sharath/easystore/code/data/'
annotations_path = os.path.join(coco_path, 'annotations', 'instances_val.json')
images_path = os.path.join(coco_path, 'images', 'val')
augmented_images_path = os.path.join(coco_path, 'images', 'augmented')
augmented_annotations_path = './augmented_dataset/instances_augmented.json'

# Ensure output directories exist
os.makedirs(augmented_images_path, exist_ok=True)
os.makedirs('./augmented_dataset', exist_ok=True)

# Load COCO annotations
coco = COCO(annotations_path)

# Helper functions
def apply_augmentations(image):
    pil_image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(degrees=10),
    ])
    return transform(pil_image)

def polygon_to_segmentation(polygon, height, width):
    rle = mask_utils.frPyObjects(polygon, height, width)
    return mask_utils.decode(rle)

def segmentation_to_polygon(segmentation):

    if len(segmentation.shape) == 2:
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)
        _, binary_mask = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif segmentation.shape[2] == 2 and len(segmentation.shape) ==3:
        # segmentation = cv2.merge([segmentation[:, :, 0], segmentation[:, :, 0], segmentation[:, :, 0]])
        contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    else:
        contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [contour.flatten().tolist() for contour in contours if len(contour) > 4]

# Augment dataset
augmented_annotations = {
    'images': [],
    'annotations': [],
    'categories': coco.dataset['categories']
}

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(images_path, img_info['file_name'])
    image = cv2.imread(img_path)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 2:
        image = cv2.merge([image[:, :, 0], image[:, :, 0], image[:, :, 0]])

    height, width = img_info['height'], img_info['width']

    # Save original and augmented images
    for i in range(2):  # 1 original + 1 augmented
        if i == 1:
            image = apply_augmentations(image)
        new_file_name = f"aug_{i}_{img_info['file_name']}"
        new_img_path = os.path.join(augmented_images_path, new_file_name)
        image =np.array(image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 2:
            image = cv2.merge([image[:, :, 0], image[:, :, 0], image[:, :, 0]])

        cv2.imwrite(new_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        new_img_id = len(augmented_annotations['images']) + 1
        augmented_annotations['images'].append({
            'id': new_img_id,
            'file_name': new_file_name,
            'height': height,
            'width': width
        })

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        for ann in annotations:
            new_ann = ann.copy()
            new_ann['id'] = len(augmented_annotations['annotations']) + 1
            new_ann['image_id'] = new_img_id

            # Update segmentation
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    segmentation = polygon_to_segmentation(ann['segmentation'], height, width)
                    new_ann['segmentation'] = segmentation_to_polygon(segmentation)
                else:
                    new_ann['segmentation'] = ann['segmentation']

            # Update bounding box
            if 'bbox' in ann:
                segmentation_mask = mask_utils.decode(mask_utils.frPyObjects(ann['segmentation'], height, width))
                x, y, w, h = cv2.boundingRect(segmentation_mask.astype(np.uint8))
                new_ann['bbox'] = [x, y, w, h]

            augmented_annotations['annotations'].append(new_ann)

# Save updated annotations
with open(augmented_annotations_path, 'w') as f:
    json.dump(augmented_annotations, f)

# COCO Viewer
def coco_viewer(original_image_path, augmented_image_path, annotations, height, width):
    original_image = cv2.imread(original_image_path)
    augmented_image = cv2.imread(augmented_image_path)

    def draw_annotations(image, annotations):
        for ann in annotations:
            # Draw bounding box
            if 'bbox' in ann:
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw mask
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.polylines(image, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
                else:
                    mask = mask_utils.decode(ann['segmentation'])
                    colored_mask = (mask * 255).astype(np.uint8)
                    image[colored_mask > 0] = [0, 255, 0]

    original_copy = original_image.copy()
    augmented_copy = augmented_image.copy()

    draw_annotations(original_copy, annotations)
    draw_annotations(augmented_copy, annotations)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(cv2.cvtColor(original_copy, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[1].imshow(cv2.cvtColor(augmented_copy, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Augmented Image")
    plt.show()

# Create PyTorch Dataset
def collate_fn(batch):
    return tuple(zip(*batch))

class CocoDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None):
        self.coco = COCO(annotations_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_id = self.coco.getImgIds()[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        bboxes = []
        for ann in anns:
            if 'segmentation' in ann:
                mask = mask_utils.decode(mask_utils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width']))
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]  # Take the first channel if multi-channel
                masks.append(mask)
            if 'bbox' in ann:
                bboxes.append(ann['bbox'])

        if self.transform:
            image = self.transform(image)

        return image, anns, masks, bboxes


# DataLoader Setup
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CocoDataset(annotations_file=augmented_annotations_path, 
                      image_dir=augmented_images_path,
                      transform=transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Example usage
for images, annotations, masks, bboxes in dataloader:
    print(f"Loaded batch of {len(images)} images with {len(masks)} masks and {len(bboxes)} bounding boxes")
