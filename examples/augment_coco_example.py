from math import ceil
from coco_augmentation.augment import COCOAugmenter
from coco_augmentation.config import TRANSFORMATIONS
from coco_augmentation.utils import visualize_augmented_images

if __name__ == "__main__":
    annotation_path = "/media/sharath/easystore/code/data/annotations/instances_val.json"
    image_dir = "/media/sharath/easystore/code/data/val/"
    output_dir = "augmented_output/"

    augmenter = COCOAugmenter(annotation_path, output_dir, TRANSFORMATIONS)
    augmenter.process(image_dir,output_dir)
    print("Augmentation complete. Check output directory.")

    # visualize_augmented_images(
    #     image_dir=output_dir,
    #     save_path="augmented_grid.jpg",
    #     grid_size=(ceil(len(augmenter.coco.getImgIds())/5), 5),
    #     show_image=False
    # )
