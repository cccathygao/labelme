import argparse
import json
import os

def labelme_to_coco(image_id, labelme_json_path, output_path):
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)
    
    output = {
        "images": [{
            "id": image_id,
            "file_path": data['imagePath'],
            "data_source": "https://huggingface.co/datasets/qixiangbupt/grefcoco",
            "height": data['imageHeight'],
            "width": data['imageWidth'],
            "scene": "", # from cmd arg
            "is_longtail": False,
            "task": "referring_segmentation",
            "problem_type": { # from cmd arg
                "num_class": "1",
                "num_instance": "N"
            }
        }],
        "annotations": []
    }
    
    for ann_id, shape in enumerate(data['shapes']):
        points = shape['points']
        
        # xs = [p[0] for p in points]
        # ys = [p[1] for p in points]
        
        # x_min, x_max = min(xs), max(xs)
        # y_min, y_max = min(ys), max(ys)
        # bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
 
        annotation = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": None,
            "bbox": None,
            "area": None,
            "segmentation": points,
            "shape_type": shape['shape_type'],
            "error_type": shape['error_type'],
            "iou": shape['iou']
        }
        
        output['annotations'].append(annotation)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

# # Usage
# labelme_to_coco('cathy_refseg_test/96857_gt.json', 'output_coco.json')


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
    
    labelme_to_coco(
        image_id=args.image_id,
        image_dir=IMAGE_DIR
    )

if __name__ == "__main__":
    main()

