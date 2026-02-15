#!/usr/bin/env python
"""
create_labelme_format_gt.py

Convert semantic segmentation groundtruth data (Cityscape) to labelme format.
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

def convert_to_labelme_format(image_id, output_dir, gt_labels):
    """
    Convert semantic segmentation groundtruth data (Cityscape) to labelme format.
    
    Args:
        image_id: Image ID (e.g., 63509)
        output_dir: Directory containing images; also directory to save gt JSON files
    """
    # Load data
    gt_json_path = osp.join(output_dir, f'{image_id}_original.json')
    print(f'Loading groundtruth json: {gt_json_path}')
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    
    image_height = data['imgHeight']
    image_width = data['imgWidth']
    objects = data['objects']
    shapes = []
    id = 0
    for item in objects:
        if item['label'] in gt_labels:
            shape = {
                "id": id,
                "label": item['label'],
                "error_type": "groundtruth",
                "points": item['polygon'],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)
            id += 1

    shapes.sort(key=lambda x:x['label'])
    
    image_path = f'{image_id}.png'
    
    # Generate imageData
    # image_data_bytes = LabelFile.load_image_file(image_path)
    # image_data_b64 = base64.b64encode(image_data_bytes).decode('utf-8')

    # Create labelme format JSON
    labelme_data = {
        "version": "labelme_semantic_seg",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "gt_labels": gt_labels,
    }
    
    # Save to file
    output_file = osp.join(output_dir, f"{image_id}_gt.json")
    with open(output_file, 'w') as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description="Convert groundturth data to labelme format GT file"
    )
    parser.add_argument(
        '--image_id',
        type=str,
        required=True,
        help='Image ID to convert (e.g., 63509)'
    )
    parser.add_argument(
        '--gt_labels',
        nargs='+',
        type=str,
        required=True,
        help='Groundtruth labels'
    )
    
    args = parser.parse_args()

    processed_labels = []
    for item in args.gt_labels:
        # Split by comma and strip any surrounding whitespace
        split_items = [x.strip() for x in item.split(',') if x.strip()]
        processed_labels.extend(split_items)
    
    args.gt_labels = processed_labels
    
    OUTPUT_DIR = 'temp/'
    
    # Convert
    try:
        convert_to_labelme_format(
            image_id=args.image_id,
            output_dir=OUTPUT_DIR,
            gt_labels=args.gt_labels
        )
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()

