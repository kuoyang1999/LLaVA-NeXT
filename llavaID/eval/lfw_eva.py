import os
import random
import setproctitle
import json
import time
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import copy

from tqdm import tqdm
import torch
import warnings
import glob

warnings.filterwarnings("ignore")

# Set the process name
setproctitle.setproctitle("LLaVA-Face")

# Set random seed for reproducibility
random.seed(42)

# Load model
base_dir = os.path.dirname(os.path.abspath(__file__))
pretrained = os.path.join(base_dir, "../../../../models/llava-onevision-qwen2-0.5b-ov")
model_name = "llava_qwen"
device = "cuda"
llava_model_args = {"multimodal": True}
overwrite_config = {"image_aspect_ratio": "pad"}
llava_model_args["overwrite_config"] = overwrite_config
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device, **llava_model_args
)
model.eval()

# Path to the lfw_funneled dataset
lfw_dir = os.path.join(base_dir, "./dataset/lfw_funneled")

# Get list of all person folders
person_folders = [
    os.path.join(lfw_dir, person)
    for person in os.listdir(lfw_dir)
    if os.path.isdir(os.path.join(lfw_dir, person))
]

# Build a list of all images with their paths
all_images = []
for person_folder in person_folders:
    images = glob.glob(os.path.join(person_folder, "*.jpg"))
    all_images.extend(images)

# Before the main loop, create/open the JSON file
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(base_dir, "./results")
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f"eval_{timestamp}.json")
existing_results = []
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        existing_results = json.load(f)

# For evaluation results
results = []

# Function to parse the model's response
def parse_multi_choice_response(response_text):
    response_text = response_text.strip()
    for char in response_text:
        if char in ['B', 'C', 'D', 'E']:
            return char
    # Default to 'E' if none of the options are found
    return 'E'

# Loop over each person folder
for person_folder in tqdm(person_folders):
    person_name = os.path.basename(person_folder)
    person_images = glob.glob(os.path.join(person_folder, "*.jpg"))
    if len(person_images) == 0:
        continue  # Skip if no images in the folder

    # Select image A
    image_A_path = random.choice(person_images)
    image_A = Image.open(image_A_path)

    # Prepare images B, C, D
    images_BCD = []
    labels_BCD = []  # True if same person as A, False otherwise
    correct_option = "E"  # Default to 'E' (none of B, C, D is same as A)

    # Remove image_A_path from person_images to avoid duplication
    remaining_person_images = [img for img in person_images if img != image_A_path]

    # If there is another image of the same person
    if len(remaining_person_images) >= 1:
        # Select another image of the target person
        image_same_person_path = random.choice(remaining_person_images)
        image_same_person = Image.open(image_same_person_path)
        # Now randomly choose a position among B, C, D to place this image
        positions = ["B", "C", "D"]
        same_person_position = random.choice(positions)
        # Prepare images_BCD list
        positions.remove(same_person_position)
        # For other positions, select images from other people
        distractor_images = []
        while len(distractor_images) < 2:
            distractor_image_path = random.choice(all_images)
            distractor_person_name = os.path.basename(
                os.path.dirname(distractor_image_path)
            )
            if (
                distractor_person_name != person_name
                and distractor_image_path not in distractor_images
            ):
                distractor_images.append(distractor_image_path)
        # Assign images to positions
        position_image_map = {
            same_person_position: image_same_person_path,
            positions[0]: distractor_images[0],
            positions[1]: distractor_images[1],
        }
        # Now, sort the images in order B, C, D
        images_BCD_paths = [position_image_map[pos] for pos in ["B", "C", "D"]]
        images_BCD = [Image.open(img_path) for img_path in images_BCD_paths]
        # Set labels_BCD
        labels_BCD = [pos == same_person_position for pos in ["B", "C", "D"]]
        # Set correct_option
        correct_option = same_person_position
    else:
        # # skip for now
        # continue
        # No other image of the same person, all images B, C, D are distractors
        distractor_images = []
        while len(distractor_images) < 3:
            distractor_image_path = random.choice(all_images)
            distractor_person_name = os.path.basename(
                os.path.dirname(distractor_image_path)
            )
            if (
                distractor_person_name != person_name
                and distractor_image_path not in distractor_images
            ):
                distractor_images.append(distractor_image_path)
        images_BCD_paths = distractor_images
        images_BCD = [Image.open(img_path) for img_path in images_BCD_paths]
        # Set labels_BCD
        labels_BCD = [False, False, False]
        correct_option = "E"  # None of B, C, D is same as A

    # Prepare the question
    images = [image_A] + images_BCD  # Images A, B, C, D
    DEFAULT_IMAGE_TOKEN = "<image>"
    question = f"{DEFAULT_IMAGE_TOKEN} This is image A.\n"
    options = ["B", "C", "D"]
    for idx, _ in enumerate(images_BCD):
        question += f"{DEFAULT_IMAGE_TOKEN} This is image {options[idx]}.\n"
    question += "In image B, image C, image D, which one is the same person as image A (answer E if none) and explain why"

    # Process images
    image_tensors = process_images(images, image_processor, model.config)
    image_tensors = [
        _image.to(dtype=torch.float16, device=device) for _image in image_tensors
    ]
    image_sizes = [image.size for image in images]

    # Prepare interleaved text-image input
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    response = text_outputs[0]
    # print(response)

    # Parse the response to get the answer
    model_answer = parse_multi_choice_response(response)
    # print("model answer:" + model_answer)

    # Record the result
    result = {
        "person_name": person_name,
        "correct_option": correct_option,
        "model_answer": model_answer,
        "question": question,
        "response": response,
        "images": {
            "A": os.path.basename(image_A_path),
            "B": os.path.basename(images_BCD_paths[0]),
            "C": os.path.basename(images_BCD_paths[1]),
            "D": os.path.basename(images_BCD_paths[2])
        }
    }
    results.append(result)
    
    # Save the current result to JSON file
    with open(results_file, 'w') as f:
        json.dump(existing_results + results, f, indent=4)

# After the loop, compute the accuracy
correct = sum(
    1 for res in results if res["model_answer"].strip() == res["correct_option"].strip()
)
total = len(results)
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")