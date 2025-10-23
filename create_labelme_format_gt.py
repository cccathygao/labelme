#!/usr/bin/env python
"""
create_labelme_format_gt.py

Convert referring segmentation dataset entry to labelme format.
Usage: python create_labelme_format_gt.py --image_id 63509
"""

import json
import base64
import os.path as osp
import argparse
import json
import re
from typing import List, Dict, Any
from PIL import Image
from labelme._label_file import LabelFile
from labelme import __version__

def parse_segmentation_string(seg_string: str) -> List[List[List[float]]]:
    """
    Parse a segmentation string and extract coordinate lists for each <seg> tag.
    
    Args:
        seg_string: String containing one or more <seg>...</seg> tags with (x,y) coordinates
        
    Returns:
        List of coordinate lists, one for each <seg> tag
    """
    # Find all <seg>...</seg> blocks
    seg_pattern = r'<seg>(.*?)</seg>'
    seg_blocks = re.findall(seg_pattern, seg_string)
    
    all_shapes_points = []
    
    for seg_block in seg_blocks:
        # Extract all (x,y) tuples from this seg block
        coord_pattern = r'\(([^,]+),([^)]+)\)'
        coords = re.findall(coord_pattern, seg_block)
        
        # Convert to list of [x, y] pairs
        points = [[float(x.strip()), float(y.strip())] for x, y in coords]
        all_shapes_points.append(points)
    
    return all_shapes_points

def convert_to_labelme_format(image_id, dataset_json_path, image_dir, output_dir):
    """
    Convert referring segmentation dataset entry to labelme format.
    
    Args:
        image_id: Image ID (e.g., 63509)
        dataset_json_path: Path to dataset.json
        image_dir: Directory containing images
        output_dir: Directory to save output JSON files
    """
    # Load dataset
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)
    
    # Find entry with matching image_id
    entry = dataset[image_id]
    
    # Extract data from entry
    problem = entry['problem']
    segstr = entry['answer']  # Should be <seg>...</seg> string
    image_path = entry['images'][0][0]
    image_height = entry.get('height')
    image_width = entry.get('width')
    
    save_path = image_id + '_gt.png'
    
    full_image_path = osp.join(image_dir, image_path)
    image_save_path = osp.join(output_dir, save_path)
    original_image = Image.open(full_image_path)
    original_image.save(image_save_path)
    print(f'cathy debug: full_image_path: {full_image_path}')
    print(f'cathy debug: image_save_path: {image_save_path}')
    
    # Generate imageData
    image_data_bytes = LabelFile.load_image_file(full_image_path)
    image_data_b64 = base64.b64encode(image_data_bytes).decode('utf-8')
    
    # Parse the segmentation string
    all_shapes_points = parse_segmentation_string(segstr)
    
    # Create shapes list
    shapes = []
    for points in all_shapes_points:
        shape = {
            "label": problem,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)
    
    # Create labelme format JSON
    labelme_data = {
        "version": __version__,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": image_data_b64,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    
    # Save to file
    output_file = osp.join(output_dir, f"{image_id}_gt.json")
    with open(output_file, 'w') as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset entry to labelme format GT file"
    )
    parser.add_argument(
        '--image_id',
        type=str,
        required=True,
        help='Image ID to convert (e.g., 63509)'
    )
    
    args = parser.parse_args()
    
    # Hardcoded paths
    IMAGE_DIR = 'cathy_refseg_test'
    OUTPUT_DIR = 'cathy_refseg_test'
    DATASET_JSON = 'dataset.json'
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert
    try:
        convert_to_labelme_format(
            image_id=args.image_id,
            dataset_json_path=DATASET_JSON,
            image_dir=IMAGE_DIR,
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()

