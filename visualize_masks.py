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

CITYSCAPE_CLASS_TO_ID = {
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

ADE_classname_to_id = {
    'wall': 1, 'building': 2, 'edifice': 2, 'sky': 3, 'floor': 4, 'flooring': 4, 'tree': 5, 
    'ceiling': 6, 'road': 7, 'route': 7, 'bed': 8, 'windowpane': 9, 'window': 9, 'grass': 10, 
    'cabinet': 11, 'sidewalk': 12, 'pavement': 12, 'person': 13, 'individual': 13, 'someone': 13, 
    'somebody': 13, 'mortal': 13, 'soul': 13, 'earth': 14, 'ground': 14, 'door': 15, 'double door': 15, 
    'table': 16, 'mountain': 17, 'mount': 17, 'plant': 18, 'flora': 18, 'plant life': 18, 
    'curtain': 19, 'drape': 19, 'drapery': 19, 'mantle': 19, 'pall': 19, 'chair': 20, 'car': 21, 
    'auto': 21, 'automobile': 21, 'machine': 21, 'motorcar': 21, 'water': 22, 'painting': 23, 
    'picture': 23, 'sofa': 24, 'couch': 24, 'lounge': 24, 'shelf': 25, 'house': 26, 'sea': 27, 
    'mirror': 28, 'rug': 29, 'carpet': 29, 'carpeting': 29, 'field': 30, 'armchair': 31, 
    'seat': 32, 'fence': 33, 'fencing': 33, 'desk': 34, 'rock': 35, 'stone': 35, 'wardrobe': 36, 
    'closet': 36, 'press': 36, 'lamp': 37, 'bathtub': 38, 'bathing tub': 38, 'bath': 38, 'tub': 38, 
    'railing': 39, 'rail': 39, 'cushion': 40, 'base': 41, 'pedestal': 41, 'stand': 41, 'box': 42, 
    'column': 43, 'pillar': 43, 'signboard': 44, 'sign': 44, 'chest of drawers': 45, 'chest': 45, 
    'bureau': 45, 'dresser': 45, 'counter': 46, 'sand': 47, 'sink': 48, 'skyscraper': 49, 
    'fireplace': 50, 'hearth': 50, 'open fireplace': 50, 'refrigerator': 51, 'icebox': 51, 
    'grandstand': 52, 'covered stand': 52, 'path': 53, 'stairs': 54, 'steps': 54, 'runway': 55, 
    'case': 56, 'display case': 56, 'showcase': 56, 'vitrine': 56, 'pool table': 57, 
    'billiard table': 57, 'snooker table': 57, 'pillow': 58, 'screen door': 59, 'screen': 59, 
    'stairway': 60, 'staircase': 60, 'river': 61, 'bridge': 62, 'span': 62, 'bookcase': 63, 
    'blind': 64, 'screen': 64, 'coffee table': 65, 'cocktail table': 65, 'toilet': 66, 'can': 66, 
    'commode': 66, 'crapper': 66, 'pot': 66, 'potty': 66, 'stool': 66, 'throne': 66, 'flower': 67, 
    'book': 68, 'hill': 69, 'bench': 70, 'countertop': 71, 'stove': 72, 'kitchen stove': 72, 
    'range': 72, 'kitchen range': 72, 'cooking stove': 72, 'palm': 73, 'palm tree': 73, 
    'kitchen island': 74, 'computer': 75, 'computing machine': 75, 'computing device': 75, 
    'data processor': 75, 'electronic computer': 75, 'information processing system': 75, 
    'swivel chair': 76, 'boat': 77, 'bar': 78, 'arcade machine': 79, 'hovel': 80, 'hut': 80, 
    'hutch': 80, 'shack': 80, 'shanty': 80, 'bus': 81, 'autobus': 81, 'coach': 81, 'charabanc': 81, 
    'double-decker': 81, 'jitney': 81, 'motorbus': 81, 'motorcoach': 81, 'omnibus': 81, 
    'passenger vehicle': 81, 'towel': 82, 'light': 83, 'light source': 83, 'truck': 84, 
    'motortruck': 84, 'tower': 85, 'chandelier': 86, 'pendant': 86, 'pendent': 86, 'awning': 87, 
    'sunshade': 87, 'sunblind': 87, 'streetlight': 88, 'street lamp': 88, 'booth': 89, 'cubicle': 89, 
    'stall': 89, 'kiosk': 89, 'television receiver': 90, 'television': 90, 'television set': 90, 
    'tv': 90, 'tv set': 90, 'idiot box': 90, 'boob tube': 90, 'telly': 90, 'goggle box': 90, 
    'airplane': 91, 'aeroplane': 91, 'plane': 91, 'dirt track': 92, 'apparel': 93, 
    'wearing apparel': 93, 'dress': 93, 'clothes': 93, 'pole': 94, 'land': 95, 'ground': 95, 
    'soil': 95, 'bannister': 96, 'banister': 96, 'balustrade': 96, 'balusters': 96, 'handrail': 96, 
    'escalator': 97, 'moving staircase': 97, 'moving stairway': 97, 'ottoman': 98, 'pouf': 98, 
    'pouffe': 98, 'puff': 98, 'hassock': 98, 'bottle': 99, 'buffet': 100, 'counter': 100, 
    'sideboard': 100, 'poster': 101, 'posting': 101, 'placard': 101, 'notice': 101, 'bill': 101, 
    'card': 101, 'stage': 102, 'van': 103, 'ship': 104, 'fountain': 105, 'conveyer belt': 106, 
    'conveyor belt': 106, 'conveyer': 106, 'conveyor': 106, 'transporter': 106, 'canopy': 107, 
    'washer': 108, 'automatic washer': 108, 'washing machine': 108, 'plaything': 109, 'toy': 109, 
    'swimming pool': 110, 'swimming bath': 110, 'natatorium': 110, 'stool': 111, 'barrel': 112, 
    'cask': 112, 'basket': 113, 'handbasket': 113, 'waterfall': 114, 'falls': 114, 'tent': 115, 
    'collapsible shelter': 115, 'bag': 116, 'minibike': 117, 'motorbike': 117, 'cradle': 118, 
    'oven': 119, 'ball': 120, 'food': 121, 'solid food': 121, 'step': 122, 'stair': 122, 
    'tank': 123, 'storage tank': 123, 'trade name': 124, 'brand name': 124, 'brand': 124, 
    'marque': 124, 'microwave': 125, 'microwave oven': 125, 'pot': 126, 'flowerpot': 126, 
    'animal': 127, 'animate being': 127, 'beast': 127, 'brute': 127, 'creature': 127, 
    'fauna': 127, 'bicycle': 128, 'bike': 128, 'wheel': 128, 'cycle': 128, 'lake': 129, 
    'dishwasher': 130, 'dish washer': 130, 'dishwashing machine': 130, 'screen': 131, 
    'silver screen': 131, 'projection screen': 131, 'blanket': 132, 'cover': 132, 'sculpture': 133, 
    'hood': 134, 'exhaust hood': 134, 'sconce': 135, 'vase': 136, 'traffic light': 137, 
    'traffic signal': 137, 'stoplight': 137, 'tray': 138, 'ashcan': 139, 'trash can': 139, 
    'garbage can': 139, 'wastebin': 139, 'ash bin': 139, 'ash-bin': 139, 'ashbin': 139, 
    'dustbin': 139, 'trash barrel': 139, 'trash bin': 139, 'fan': 140, 'pier': 141, 'wharf': 141, 
    'wharfage': 141, 'dock': 141, 'crt screen': 142, 'plate': 143, 'monitor': 144, 
    'monitoring device': 144, 'bulletin board': 145, 'notice board': 145, 'shower': 146, 
    'radiator': 147, 'glass': 148, 'drinking glass': 148, 'clock': 149, 'flag': 150,
    'dam': 151, 'cliff': 152, 'carriage': 153, 'skeleton': 154, 'drawing': 155, 'ladder': 156,
    'bandstand': 157, 'fish': 158, 'jacket': 159, 'jar': 160
}

def _get_rgb_by_label(label: str) -> tuple[int, int, int]:

    # if label not in CITYSCAPE_CLASS_TO_ID:
    #     print(f"Warning: Unknown label '{label}', using default color")
    #     index = 0
    # else:
    #     index = CITYSCAPE_CLASS_TO_ID[label]

    if label not in ADE_classname_to_id:
        print(f"Warning: Unknown label '{label}', using default color")
        index = 0
    else:
        index = ADE_classname_to_id[label]

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
    
    OUTPUT_DIR = 'output_masks'
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