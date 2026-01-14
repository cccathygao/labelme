import json
import os
import argparse

def load_from_json(json_path='results.json'):
    """Load existing results or create empty list"""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return []

def save_to_json(results, json_path='results.json'):
    """Save results to JSON file"""
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved result to ", json_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate referring segmentation test cases')
    parser.add_argument('--image_id', type=int, required=True,
                        help='Process a specific image ID')
    parser.add_argument('--output_path', type=str,
                        help='Path to the output json file')
    parser.add_argument('--eval_type', type=str, required=True,
                        help='Evaluation type: pointwise, pairwise, listwise')
    
    args = parser.parse_args()
    
    return args

def generate_pairwise_test(image_id, output_path):

    input_path = f'coco_format/{image_id}.json'
    image_data = load_from_json(input_path)
    test_cases = load_from_json(output_path)
    
    CRITERIA = 'The criteria for judging are: '
    CRITERIA += '1. **Completeness:** Does the mask cover the *entire* object/region described by the referring expression? '
    CRITERIA += '2. **Accuracy:** Does the mask strictly adhere to the boundaries of the described object/region? (Minimize false positives, i.e., covering background or other objects.) '
    CRITERIA += '3. **Ambiguity Resolution:** If the expression is ambiguous (e.g., "the largest red ball"), does the mask correctly select the intended instance based on visual context? '
    CRITERIA += '4. **Linguistic Alignment:** Does the mask correctly interpret all descriptive elements (color, size, position, action, relation to other objects) in the expression?'

    task_desc = 'You are an expert vision-language evaluator. '
    task_desc += 'Your task is to perform a pairwise comparison to determine which segmentation mask '
    task_desc += 'best matches the given referring expression.'
    task_desc += 'The segmenation mask is highlighted in red color in each image.'

    # TODO: add original image
    
    problem = image_data['images'][0]['problem']
    problem_type = image_data['images'][0]['problem_type']
    masks = image_data['annotations']
    
    OUTPUT_DIR = 'grefcoco_masks'
    
    for i in range(len(masks)):
        entry_1 = masks[i]
        mask_id1 = entry_1['id']
        image_path_1 = os.path.join(OUTPUT_DIR, f"{image_id}_mask_{mask_id1}.png")
        e1 = entry_1.get('error_type', '')
        iou_1 = entry_1.get('iou', 0.0)
        
        for j in range(i+1, len(masks)):
            entry_2 = masks[j]
            mask_id2 = entry_2['id']
            image_path_2 = os.path.join(OUTPUT_DIR, f"{image_id}_mask_{mask_id2}.png")
            e2 = entry_2.get('error_type', '')
            iou_2 = entry_2.get('iou', 0.0)
        
            if ((e1 == 'groundtruth' or e2 == 'groundtruth')
            or (e1 == e2 and iou_1 != iou_2)):
                answer = ''
                if e1 == 'groundtruth':
                    answer = 'A'
                elif e2 == 'groundtruth':
                    answer = 'B'
                elif iou_1 > iou_2:
                    answer = 'A'
                else:
                    answer = 'B'
                new_test_case = {
                    "image_id": image_id,
                    "prompt": f"{task_desc} {CRITERIA} Given the referring expression: {problem}, which generated image is better?\nA. <image>\nB. <image>\nPlease answer with A or B directly.",
                    "media": [image_path_1, image_path_2],
                    "choices": ['A','B'],
                    "answer": answer,
                    "metadata": {
                        "task": "referring-segmentation",
                        "referring expression": problem,
                        "score_good": max(iou_1, iou_2),
                        "score_bad": min(iou_1, iou_2),
                        "score_difference": abs(iou_1 - iou_2),
                        "error_type": next(iter({e1, e2} - {"groundtruth"})),
                        "problem_type": problem_type
                    },
                    "comment": f"Image1 - Label: {e1}, IoU: {iou_1}. Image2 - Label: {e2}, IoU: {iou_2}"
                }
        
                test_cases.append(new_test_case)

    save_to_json(test_cases, output_path)

def main():
    args = parse_args()
    
    # single image - scoring
    # pairwise - comparison
    # list - ranking
    
    # if args.eval_type == "pointwise":
    #     output_path = args.output_path if args.output_path else "pointwise_tests.json"
    #     generate_pointwise_test(args.image_id, output_path)
    if args.eval_type == "pairwise":
        output_path = args.output_path if args.output_path else "pairwise_tests.json"
        generate_pairwise_test(args.image_id, output_path)
    # elif args.eval_type == "listwise":
    #     output_path = args.output_path if args.output_path else "listwise_tests.json"
    #     generate_listwise_test(args.image_id, output_path)  
        

if __name__ == "__main__":
    main()
    
# Usage:
# python generate_test.py --eval_type pairwise --output_path test_bed/pairwise_test.json --image_id 63509
# python generate_test.py --eval_type listwise --output_path test_bed/listwise_test.json --image_id 63509
# python generate_test.py --eval_type pointwise --output_path test_bed/pointwise_test.json --image_id 63509