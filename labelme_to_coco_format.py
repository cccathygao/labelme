import argparse
import json
import os

def labelme_to_coco(image_id, scene, num_class, num_instance):
    labelme_json_path = f"output/{image_id}.json"
    
    if not os.path.exists(labelme_json_path):
        raise FileNotFoundError(f"File not found: {labelme_json_path}")
    
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)
    
    output = {
        "images": [{
            "id": image_id,
            "file_path": data['imagePath'].replace("_gt", ""),
            "data_source": "https://huggingface.co/datasets/qixiangbupt/grefcoco",
            "height": data['imageHeight'],
            "width": data['imageWidth'],
            "scene": scene,
            "is_crowd": False,
            "is_longtail": False,
            "task": "referring_segmentation",
            "problem": data['problem'],
            "problem_type": {
                "num_class": num_class,
                "num_instance": num_instance
            }
        }],
        "annotations": []
    }
    
    id_to_shapes = {}
    
    # single shape
    for ann_id, shape in enumerate(data['shapes']):
        shape_id = shape['id']
        id_to_shapes[shape_id] = shape
        
        points = shape['points']
        
        # xs = [p[0] for p in points]
        # ys = [p[1] for p in points]
        
        # x_min, x_max = min(xs), max(xs)
        # y_min, y_max = min(ys), max(ys)
        # bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        
        error_type = None if shape['error_type'] == data['problem'] else shape['error_type']
        iou = 1.0 if shape['error_type'] == data['problem'] else shape['iou']
 
        annotation = {
            "id": ann_id,
            "image_id": image_id,
            "class_id": None,
            "bbox": None,
            "area": None,
            "shape_type": shape['shape_type'],
            "error_type": error_type,
            "iou": iou,
            "segmentation": [points] # list[list[list[float]]]
        }
        
        output['annotations'].append(annotation)
    
    ann_id = len(data['shapes'])
     
    # multi shapes
    for item in data['combinedShapes']:
        ids = item['ids']
        error_type = item['error_type']
        iou = item['iou']
        
        points = []
        for id in ids:
            assert(id == id_to_shapes[id]['id'])
            current_shape_points = id_to_shapes[id]['points']
            points.append(current_shape_points)
        
        annotation = {
            "id": ann_id,
            "image_id": image_id,
            "class_id": None,
            "bbox": None,
            "area": None,
            "shape_type": "polygon",
            "error_type": error_type,
            "iou": iou,
            "segmentation": points # list[list[list[float]]]
        }
        
        ann_id += 1
        
        output['annotations'].append(annotation)
    
    output_path = f"coco_format/{image_id}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

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
    parser.add_argument(
        '--scene',
        type=str,
        required=True,
        help='Scene description'
    )
    parser.add_argument(
        '--cls',
        type=str,
        required=True,
        help='Num class'
    )
    parser.add_argument(
        '--ins',
        type=str,
        required=True,
        help='Num instance'
    )
    
    args = parser.parse_args()
    
    labelme_to_coco(
        image_id=args.image_id,
        scene=args.scene, 
        num_class=args.cls, 
        num_instance=args.ins
    )

if __name__ == "__main__":
    main()

