
# ----------------------------------------------------------------------------#

from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

# ----------------------------------------------------------------------------#

image = '.venv/model_test.jpg'
output_dir = '/home/ml3/Desktop/CRAFT/.venv/outputs'

image = read_image(image)

refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)

# ----------------------------------------------------------------------------#

# Predictions.
prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=True,
    long_size=1280
)

# Detected text regions.
exported_file_paths = export_detected_regions(
    image=image,
    regions=prediction_result["boxes"],
    output_dir=output_dir,
    rectify=True
)

# Heatmap, detection points and box visualization.
export_extra_results(
    image=image,
    regions=prediction_result["boxes"],
    heatmaps=prediction_result["heatmaps"],
    output_dir=output_dir
)

# Unloading the models from gpu.
empty_cuda_cache()

# ----------------------------------------------------------------------------#
