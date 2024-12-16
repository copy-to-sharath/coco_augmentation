import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os


import matplotlib.pyplot as plt
import numpy as np


class COCOAugmenter:
    def __init__(self, annotation_path, output_dir, transformations):
        """
        COCOAugmenter handles augmentations for COCO-format datasets, supporting both polygons and RLE masks.
        """
        self.coco = COCO(annotation_path)
        self.output_dir = output_dir
        self.transform = A.Compose(
            transformations, additional_targets={"mask": "mask"}, bbox_params=dict(
            format='coco',
            min_visibility=0.0
            ),
        )
        os.makedirs(output_dir, exist_ok=True)

    def load_image_and_mask(self,image_dir, image_id):
        """
        Loads an image and its corresponding segmentation mask (polygon or RLE).
        """

        image_info = self.coco.loadImgs(image_id)[0]
        image = cv2.imread(os.path.join(image_dir,image_info['file_name']))
        h, w = image.shape[:2]
        file_name =image_info['file_name']
        mask = np.zeros((h, w), dtype=np.uint8)
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        bboxes = [[item["bbox"][0], item["bbox"][1],item["bbox"][2], item["bbox"][3], item["category_id"]] for item in annotations]
        color_index =-1    

        if len(annotations) > 0:
            N = len(annotations)
            colours = plt.cm.get_cmap('viridis', N)  # Change the string from 'viridis' to whatever you want from the above link
            cmap = (colours(np.linspace(0, 1, N)) * 255)  # Obtain RGB colour map
            cmap[0,-1] = 0  # Set alpha for label 0 to be 0
            cmap[1:,-1] = 0.3  # Set the other alphas for the labels to be 0.3

            for ann in annotations:
                color_index+=1
                if 'segmentation' in ann:


                    if isinstance(ann['segmentation'], list):  # Polygon

                        for seg in ann['segmentation']:
                            poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                            # Get the color for a specific class (e.g., class 0)
                            # print(color)
                            cv2.fillPoly(mask, [poly], color=cmap[color_index])

                    elif isinstance(ann['segmentation'], dict):  # RLE
                        rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                        binary_mask = maskUtils.decode(rle)
                        mask = np.maximum(mask, binary_mask)  # Combine masks for all objects

        return image, bboxes, mask, annotations,file_name

    def augment_image_and_mask(self, image, bboxes,masks):
        """
        Applies augmentations to the image and mask.
        """
        augmented = self.transform(image=image, mask=masks,bboxes=bboxes)
        return augmented["image"], augmented["bboxes"],augmented["mask"]

    def extract_polygons_from_mask(self, mask):
        """
        Extracts polygons from the augmented binary mask.
        """
        polygons = []

        if not isinstance(mask,np.ndarray):
            mask =mask.numpy().astype(np.uint8)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) >= 6:  # Valid polygons only
                contour = contour.reshape(-1, 2)
                polygon = contour.flatten().tolist()
                polygons.append(polygon)

        return polygons

    def save_augmented_data(self, augmented_image, annotations, index):
        """
        Saves the augmented image.
        """
        output_image_path = os.path.join(self.output_dir, f"aug_image_{index}.jpg")

        if augmented_image.is_cuda:
            augmented_image = augmented_image.cpu().numpy()

        # OpenCV expects BGR format, so convert if needed
        if augmented_image.shape[0] == 3:
            augmented_image = augmented_image.permute(1, 2, 0)  # Convert to HWC format        

        # Scale the values from [0, 1] to [0, 255] if needed
        if augmented_image.max() <= 1:
            augmented_image = (augmented_image * 255).astype(np.uint8)
        augmented_image  = augmented_image.numpy()
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, augmented_image)
        return output_image_path

    def process(self,image_dir,output_dir):
        """
        Main function to process all images in the dataset.
        """
        running_idx = 10000000 
        
        aug_dir_image_path =image_dir.rsplit("/")[-2]

        os.makedirs(os.path.join(output_dir,'annotations'),exist_ok=True)
        os.makedirs(os.path.join(output_dir,aug_dir_image_path),exist_ok=True)

        
        json_dict = {
                "images": [],
                "categories": self.coco.cats,
                "annotations": []
            }

        for idx, image_id in enumerate(self.coco.getImgIds()):
            running_idx = running_idx + idx
            
            image, bboxes, masks, annotations,file_name = self.load_image_and_mask(image_dir=image_dir,image_id=image_id)
            cnt =-1
            json_dict =self.generate_coco_file(image_id,json_dict, aug_dir_image_path, idx, cnt, annotations, file_name, image, bboxes, masks,False)                
            augmented_image,aug_bboxes, augmented_mask = self.augment_image_and_mask(image, bboxes,masks)

            json_dict = self.generate_coco_file(running_idx, json_dict,aug_dir_image_path, idx, cnt, annotations, file_name, augmented_image, aug_bboxes, augmented_mask,True)                

        # Write COCO json file

        
        with open(os.path.join(output_dir,'annotations',f'{image_dir.rsplit("/")[-2]}.json'), 'w') as outfile:
            outfile.write(json.dumps(json_dict, indent=4)) 

    def generate_coco_file(self, running_idx,json_dict, aug_dir_image_path, idx, cnt, annotations, file_name, augmented_image, aug_bboxes, augmented_mask, append_aug):
        
        polygons = self.extract_polygons_from_mask(augmented_mask)
        for i, polygon in enumerate(polygons):
            if polygon and i in annotations:
                annotations[i]["segmentation"] = polygon

        file_contents =file_name.rsplit(".")

        if append_aug:
            aug_file_name =f"{file_contents[0]}_{idx}_aug.{file_contents[1]}"
        else:    
            aug_file_name =f"{file_contents[0]}_{idx}.{file_contents[1]}"

        output_image_path = os.path.join(self.output_dir,aug_dir_image_path, aug_file_name)

        if not isinstance(augmented_image,np.ndarray):
            augmented_image = augmented_image.cpu().numpy()
        
            # OpenCV expects BGR format, so convert if needed
        if augmented_image.shape[0] == 3:
            if not isinstance(augmented_image,np.ndarray):
                augmented_image = augmented_image.permute(1, 2, 0)  # Convert to HWC format                        
            else:       
                augmented_image =np.transpose(augmented_image, (1, 2, 0))


            # Scale the values from [0, 1] to [0, 255] if needed
        if augmented_image.max() <= 1:
            augmented_image = (augmented_image * 255).astype(np.uint8)
        
       
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, augmented_image)

            # self.save_augmented_data(augmented_image, annotations, idx)
                
        image_object = {
                "width": augmented_image.shape[1],
                "height": augmented_image.shape[0],
                "id": running_idx,
                "file_name": aug_file_name,
                "full_path": output_image_path
            }

        json_dict["images"].append(image_object) # Update COCO json with image object

        for boxes in (aug_bboxes):
            cnt += 1 # update the object counter

            category_id = boxes[4]
            bbox = boxes[:4]
            
            coco_object = {
                    "id": cnt,
                    "image_id": running_idx,
                    "category_id": category_id,
                    "segmentation": annotations[cnt]["segmentation"],
                    "bbox": bbox,
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": boxes[2] * boxes[3]
                }

            json_dict["annotations"].append(coco_object) # update the detected object information
        running_idx += 1           

        return json_dict