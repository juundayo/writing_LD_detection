
# ----------------------------------------------------------------------------#

import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
from craft_utils import getDetBoxes

# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    # Takes the Image Dump folder as an input.
    # dump = sys.argv[1]
    dump = "/home/ml3/Desktop/Thesis/LetterDump"
    output_dir = '/home/ml3/Desktop/CRAFT/.venv/outputs'

    char_coord_dir = os.path.join(output_dir, 'char_coordinates')
    os.makedirs(char_coord_dir, exist_ok=True)

    #for im in os.listdir(dump+"/"):
    #im = "/home/ml3/Desktop/CRAFT/.venv/model_test.jpg"
    im = "/home/ml3/Desktop/Thesis/LetterDump/line0_word0.png"
    image = read_image(im)

    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, 
                               cv2.BORDER_CONSTANT, value=[255,255,255])

    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)

    # Predictions.
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.5,
        link_threshold=0.7,
        low_text=0.2,
        cuda=True,
        long_size=1280
    )

    # Detected text regions.
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["polys"],
        output_dir=output_dir,
        rectify=True
    )

    # Heatmap, detection points and box visualization.
    export_extra_results(
        image=image,
        regions=prediction_result["polys"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=output_dir
    )

    score_text = prediction_result["heatmaps"]["text_score_heatmap"]
    score_link = prediction_result["heatmaps"]["link_score_heatmap"]
    
    textmap = np.array(score_text)
    linkmap = np.array(score_link)

    if len(textmap.shape) == 3:
        textmap = textmap[:,:,0]
    if len(linkmap.shape) == 3:
        linkmap = linkmap[:,:,0]

    textmap = textmap.astype(np.float32) / 255.0
    linkmap = linkmap.astype(np.float32) / 255.0

    cv2.imwrite(os.path.join(output_dir, 'text_heatmap.png'), (textmap * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'link_heatmap.png'), (linkmap * 255).astype(np.uint8))
    
    char_boxes, _ = getDetBoxes(textmap,
                                linkmap,
                                text_threshold=0.5,
                                link_threshold=0.7,
                                low_text=0.2)
    
    char_boxes = np.array(char_boxes)
    base_name = os.path.splitext(os.path.basename(im))[0]
    coord_output_path = os.path.join(char_coord_dir, f"{base_name}_char_coords.json")
    
    with open(coord_output_path, 'w') as f:
        json.dump(char_boxes.tolist(), f)

    print(f"Found {len(char_boxes)} character boxes in {image}")

    for i, box in enumerate(char_boxes):
        print(f"Character {i+1} coordinates: {box.tolist()}")
    
    # Unloading the models from gpu.
    empty_cuda_cache()
    
# ----------------------------------------------------------------------------#
