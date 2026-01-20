import argparse
import json
import os

def labelme_to_coco(image_id, scene, num_class, groundtruth_labels):
    cityscapes_id_to_class = {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
        5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
        9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
        255: 'ignore'
    }

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

    labelme_json_path = f"temp/{image_id}.json"
    
    if not os.path.exists(labelme_json_path):
        raise FileNotFoundError(f"File not found: {labelme_json_path}")
    
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)
    
    output = {
        "images": [{
            "id": image_id,
            "file_path": data['imagePath'].replace("_gt", ""),
            "data_source": "https://www.cityscapes-dataset.com/",
            "height": data['imageHeight'],
            "width": data['imageWidth'],
            "scene": scene,
            "is_crowd": False,
            "is_longtail": False,
            "task": "semantic_segmentation",
            "problem_type": {
                "num_class": num_class,
            },
            "groundtruth_labels": groundtruth_labels
        }],
        "annotations": []
    }
    
    id_to_shapes = {}
    
    # single shape
    for ann_id, shape in enumerate(data['shapes']):
        shape_id = shape['id']
        id_to_shapes[shape_id] = shape
        
        # points = shape['points']
        
        # xs = [p[0] for p in points]
        # ys = [p[1] for p in points]
        
        # x_min, x_max = min(xs), max(xs)
        # y_min, y_max = min(ys), max(ys)
        # bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        
        # error_type = shape['error_type']
        # iou = shape['iou']
        
        # if shape['error_type'] == data['problem']:
        #     if iou == 1.0:
        #         error_type = 'groundtruth'
        #     else:
        #         error_type = 'under-coverage'
 
        # annotation = {
        #     "id": ann_id,
        #     "image_id": image_id,
        #     "class_id": None,
        #     "bbox": None,
        #     "area": None,
        #     "shape_type": shape['shape_type'],
        #     "error_type": error_type,
        #     "iou": iou,
        #     "segmentation": [points] # list[list[list[float]]]
        # }
        
        # output['annotations'].append(annotation)
    
    # ann_id = len(data['shapes'])
    ann_id = 0
     
    # multi shapes
    for item in data['combinedShapes']:
        error_type = item['error_type']
        mean_iou = item['mean_iou']

        if mean_iou == 1.0:
            error_type = 'groundtruth'
        
        segmentation_by_label = []
        class_ids, class_names = [], []

        results_by_label = item['results_by_label']
        for label, mask in results_by_label.items():
            class_ids.append(cityscapes_class_to_id[label])
            class_names.append(label)
            segmentation = {}
            class_iou = mask["iou"]
            ids = mask['shape_ids']
            num_shapes = mask['num_shapes']
            assert(num_shapes == len(ids))
            points = []
            for id in ids:
                assert(id == id_to_shapes[id]['id'])
                current_shape_points = id_to_shapes[id]['points']
                points.append(current_shape_points)
            segmentation['label'] = label
            segmentation['class_iou'] = class_iou
            segmentation['points'] = points # list[list[list[float]]]
            segmentation_by_label.append(segmentation)

        
        annotation = {
            "id": ann_id,
            "image_id": image_id,
            "class_id": class_ids,
            "class_names": class_names,
            "bbox": None,
            "area": None,
            "shape_type": "polygon",
            "error_type": error_type,
            "mean_iou": mean_iou,
            "segmentations": segmentation_by_label
        }
        
        ann_id += 1
        
        output['annotations'].append(annotation)
    
    output_path = f"coco_format/{image_id}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    print("Saved coco format output to ", output_path)

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
        default='outdoor-urban',
        help='Scene description'
    )
    parser.add_argument(
        '--cls',
        type=str,
        required=True,
        help='Num class'
    )
    parser.add_argument(
        '--gt_labels',
        nargs='+',  # Accepts one or more values
        type=str,   # Each value will be converted to int
        required=True,
        help='Groundtruth class names'
    )
    
    args = parser.parse_args()
    
    labelme_to_coco(
        image_id=args.image_id,
        scene=args.scene, 
        num_class=args.cls,
        groundtruth_labels=args.gt_labels
    )

if __name__ == "__main__":
    main()

