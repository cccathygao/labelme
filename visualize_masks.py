import os
import cv2
import supervision as sv
import json
import numpy as np
import imgviz
from PIL import Image
from PIL import Image, ImageDraw
from numpy.typing import NDArray

import argparse

LABEL_COLORMAP: NDArray[np.uint8] = imgviz.label_colormap()

def _get_rgb_by_label(label: str) -> tuple[int, int, int]:
    cityscapes_class_to_id = {
    'road': 0,
    'sidewalk': 1,
    'building': 2,
    'wall': 3,
    'fence': 4,
    'pole': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'vegetation': 8,
    'terrain': 9,
    'sky': 10,
    'person': 11,
    'rider': 12,
    'car': 13,
    'truck': 14,
    'bus': 15,
    'train': 16,
    'motorcycle': 17,
    'bicycle': 18,
    'ignore': 255
    }

    if label not in cityscapes_class_to_id:
        print(f"Warning: Unknown label '{label}', using default color")
        index = 0
    else:
        index = cityscapes_class_to_id[label]

    label_id: int = (
        1  # skip black color by default
        + index
    )
    rgb: tuple[int, int, int] = tuple(
        LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)].tolist()
    )
    return rgb

def prepare_image_rgb(image_input):
    """Convert image input to RGB numpy array"""
    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert('RGB'))
    elif isinstance(image_input, str):
        image_bgr = cv2.imread(image_input)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError(f"Expected PIL Image or file path, got {type(image_input)}")

def prepare_image_bgr(image_input):
    """Convert image input to BGR numpy array"""
    if isinstance(image_input, Image.Image):
        image_rgb = np.array(image_input.convert('RGB'))
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, str):
        return cv2.imread(image_input)
    else:
        raise TypeError(f"Expected PIL Image or file path, got {type(image_input)}")

def get_unique_filepath(base_path):
    """
    Generate a unique filepath by appending numbers if file exists
    
    Example:
        temp/1/mask.png -> temp/1/mask_0.png
        temp/1/mask.png -> temp/1/mask_1.png
        temp/1/mask.png -> temp/1/mask_2.png
    """

    # Split into directory, filename, and extension
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    # Try appending numbers until we find a unique name
    counter = 0
    while True:
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def coords_to_union_mask(all_masks_coords, img_width, img_height):
    # Create blank mask
    mask_img = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask_img)
    
    # Draw each polygon
    for coords in all_masks_coords:
        if len(coords) > 0:
            draw.polygon(coords, outline=1, fill=1)
    
    # Convert to numpy boolean array
    mask = np.array(mask_img, dtype=bool)
    
    return mask
       
def visualize_mask(image_id, mask_id, img_height, img_width, segmentations, output_dir):
    """Visualize the selected mask on the image"""
    
    # image input path - hardcoded
    image_input = f'temp/{image_id}.png'
    
    # Get image in BGR format
    image_bgr = prepare_image_bgr(image_input)
    annotated_image = image_bgr.copy()

    for segmentation in segmentations:
        label = segmentation['label']
        points = segmentation['points']
    
        mask = coords_to_union_mask(points, img_width, img_height)
        mask = np.expand_dims(mask, axis=0)
    
        # Use label-specific color to paint mask
        r, g, b = _get_rgb_by_label(label)
        label_color = sv.Color(r=r, g=g, b=b)
        mask_annotator = sv.MaskAnnotator(color=label_color, color_lookup=sv.ColorLookup.INDEX)
    
        # Create detections from mask
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=mask),
            mask=mask
        )
    
        # Annotate the image with mask
        annotated_image = mask_annotator.annotate(
            scene=annotated_image,  # Use the accumulated image
            detections=detections
        )

    output_path = os.path.join(output_dir, f"{image_id}_mask_{mask_id}.png")
    
    # Save directly using cv2 (preserves exact dimensions)
    cv2.imwrite(output_path, annotated_image)
    
    print(f"✓ Saved visualization to: {output_path}")
    
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description='Process grefcoco dataset with SAM')
    parser.add_argument('--image_id', type=str, required=True,
                        help='Process a specific image ID')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    OUTPUT_DIR = 'cityscape_masks'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    INPUT_JSON = f'coco_format/{args.image_id}.json'
    
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    img_height = data['images'][0]['height']
    img_width = data['images'][0]['width']
    
    for i in range(len(data['annotations'])):
        mask_id = data['annotations'][i]['id']
        segmentations = data['annotations'][i]['segmentations']
        visualize_mask(args.image_id, mask_id, img_height, img_width, segmentations, OUTPUT_DIR)
    
if __name__ == "__main__":
    main()
    

# usage: python visualize_masks.py --image_id=63509