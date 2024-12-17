# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.RandomCrop(width=256, height=256, p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# %%

from pycocotools.coco import COCO
import cv2
import numpy as np
import json

coco = COCO("/media/sharath/easystore/code/data/annotations/instances_val.json")

def load_image(image_path):
    return cv2.imread(image_path)

def load_mask(coco, image_id):
    masks = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
        mask = coco.annToMask(ann)
        masks.append(mask)
    masks = np.stack(masks, axis=-1)
    return masks

def load_bboxes(coco, image_id):
    bboxes = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
        bbox = ann['bbox']
        bboxes.append(bbox)
    return bboxes

def transform_data(image, masks, bboxes, category_ids):
    augmented = transform(image=image, masks=masks, bboxes=bboxes, category_ids=category_ids)
    image = augmented['image']
    masks = augmented['masks']
    bboxes = augmented['bboxes']
    category_ids = augmented['category_ids']
    return image, masks, bboxes, category_ids

# %%

import os

image_dir = '/media/sharath/easystore/code/data/val/'
output_json = 'augmented_output/val.json'

coco_output = {
    "images": [],
    "annotations": [],
    "categories": coco.loadCats(coco.getCatIds())
}

annotation_id = 1

for image_id in coco.imgs.keys():
    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(image_dir, image_info['file_name'])
    image = load_image(image_path)
    masks = load_mask(coco, image_id)
    bboxes = load_bboxes(coco, image_id)
    category_ids = [ann['category_id'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id))]

    # Transform data
    transformed_image, transformed_masks, transformed_bboxes, transformed_category_ids = transform_data(image, masks, bboxes, category_ids)

    # Add original image and annotations
    coco_output["images"].append({
        "id": image_id,
        "file_name": image_info['file_name'],
        "height": image_info['height'],
        "width": image_info['width']
    })

    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
        ann_id = annotation_id
        annotation_id += 1
        coco_output["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann['category_id'],
            "segmentation": ann['segmentation'],
            "bbox": ann['bbox'],
            "area": ann['area'],
            "iscrowd": ann['iscrowd']
        })

    # Add transformed image and annotations
    transformed_image_id = f"{image_id+100000000}"
    coco_output["images"].append({
        "id": transformed_image_id,
        "file_name": f"transformed_{image_info['file_name']}",
        "height": transformed_image.shape[1],
        "width": transformed_image.shape[2]
    })

    for i, transformed_bbox in enumerate(transformed_bboxes):
        ann_id = annotation_id
        annotation_id += 1
        coco_output["annotations"].append({
            "id": ann_id,
            "image_id": transformed_image_id,
            "category_id": transformed_category_ids[i],
            "segmentation": [],  # Add segmentation if available
            "bbox": transformed_bbox,
            "area": transformed_bbox[2] * transformed_bbox[3],
            "iscrowd": 0  # Update this value if needed
        })

with open(output_json, 'w') as f:
    json.dump(coco_output, f)

# %%
