import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np

def save_coco_annotations(output_path, images, annotations, categories):
    """
    Saves updated COCO annotations to a JSON file.
    """
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=4)

def visualize_augmented_images(image_dir, save_path=None, grid_size=(5, 5), show_image=False):
    """
    Visualizes augmented images in a grid and optionally saves the grid as an image.
    """
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    image_paths = image_paths[:grid_size[0] * grid_size[1]]

    images = [cv2.imread(img_path) for img_path in image_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx])
            ax.axis("off")
        else:
            ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Grid image saved to: {save_path}")
    if show_image:
        plt.show()
