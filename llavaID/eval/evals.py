import os
import random
import setproctitle
import json
import time
import argparse
import copy
import glob
import warnings
from PIL import Image

from tqdm import tqdm
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description='LLaVA Face Evaluation')
    parser.add_argument('--eval_dataset', type=str, choices=['lfw', 'cplfw', 'calfw'], default='lfw',
                        help='Dataset to evaluate on: lfw (Labeled Faces in the Wild), cplfw (Cross-Pose LFW), or calfw (Cross-Age LFW)')
    parser.add_argument('--eval_mode', type=str, choices=['pairs', 'multiple-choice', 'multiple-choice-none-option'], default='pairs',
                        help='Evaluation mode: pairs or multiple-choice')
    parser.add_argument('--model_size', type=str, choices=['0.5b', '7b'], default='7b',
                        help='Size of LLaVA model to use (0.5b or 7b)')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Number of samples to evaluate. Default -1 means evaluate all samples.')
    return parser.parse_args()


def load_lfw_dataset(lfw_dir):
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
    return person_folders, all_images


def setup_results_file(base_dir, args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(base_dir, f"./results/llava-{args.model_size}/{args.eval_mode}")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{timestamp}.json")
    existing_results = []
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
    return results_file, existing_results


def create_pairs_data(person_folders, all_images):
    data_samples = []
    for person_folder in tqdm(person_folders, desc="Creating pairs data"):
        person_name = os.path.basename(person_folder)
        person_images = glob.glob(os.path.join(person_folder, "*.jpg"))
        if len(person_images) < 2:
            continue  # Skip if less than 2 images in the folder

        # Create a positive pair
        target_image = random.choice(person_images)
        choice_image = random.choice([img for img in person_images if img != target_image])
        data_samples.append({
            'person_name': person_name,
            'target_image': target_image,
            'choices': [choice_image],
            'labels': [True],
            'correct_option': 'True'
        })

        # Create a negative pair
        choice_image = None
        while True:
            candidate_image = random.choice(all_images)
            distractor_person_name = os.path.basename(os.path.dirname(candidate_image))
            if distractor_person_name != person_name:
                choice_image = candidate_image
                break
        data_samples.append({
            'person_name': person_name,
            'target_image': target_image,
            'choices': [choice_image],
            'labels': [False],
            'correct_option': 'False'
        })
    random.shuffle(data_samples)
    return data_samples


def create_multiple_choice_data(person_folders, all_images, eval_mode):
    data_samples = []
    for person_folder in tqdm(person_folders, desc="Creating multiple-choice data"):
        person_name = os.path.basename(person_folder)
        person_images = glob.glob(os.path.join(person_folder, "*.jpg"))
        if len(person_images) == 0:
            continue  # Skip if no images in the folder

        # Select target image
        target_image = random.choice(person_images)

        # Remove target_image from person_images to avoid duplication
        remaining_person_images = [img for img in person_images if img != target_image]

        # Initialize variables
        choices = []
        labels = []  # True if same person as target_image
        correct_option = "E"  # Default to 'E' (none of the choices is same as target_image)

        # If there is another image of the same person
        if len(remaining_person_images) >= 1:
            # Select another image of the target person
            same_person_image = random.choice(remaining_person_images)
            # Now randomly choose a position among choices to place this image
            positions = [0, 1, 2]
            same_person_position = random.choice(positions)
            # Prepare choices
            positions.remove(same_person_position)
            # For other positions, select images from other people
            distractor_images = []
            while len(distractor_images) < 2:
                distractor_image = random.choice(all_images)
                distractor_person_name = os.path.basename(os.path.dirname(distractor_image))
                if distractor_person_name != person_name and distractor_image not in distractor_images:
                    distractor_images.append(distractor_image)
            # Assign images to positions
            choices_list = [None, None, None]
            choices_list[same_person_position] = same_person_image
            choices_list[positions[0]] = distractor_images[0]
            choices_list[positions[1]] = distractor_images[1]
            choices = choices_list
            # Set labels
            labels = [i == same_person_position for i in range(3)]
            # Set correct_option
            correct_option = ['B', 'C', 'D'][same_person_position]
        else:
            if eval_mode == 'multiple-choice':
                continue
            # No other image of the same person, all choices are distractors
            distractor_images = []
            while len(distractor_images) < 3:
                distractor_image = random.choice(all_images)
                distractor_person_name = os.path.basename(os.path.dirname(distractor_image))
                if distractor_person_name != person_name and distractor_image not in distractor_images:
                    distractor_images.append(distractor_image)
            choices = distractor_images
            # Set labels
            labels = [False, False, False]
            correct_option = "E"  # None of B, C, D is same as target image

        data_samples.append({
            'person_name': person_name,
            'target_image': target_image,
            'choices': choices,
            'labels': labels,
            'correct_option': correct_option
        })
    return data_samples


def generate_prompt(eval_mode, num_choices):
    DEFAULT_IMAGE_TOKEN = "<image>"
    if eval_mode == 'pairs':
        question = f"{DEFAULT_IMAGE_TOKEN}\n"
        question += f"{DEFAULT_IMAGE_TOKEN}\n"
        question += "Are these two images of the same person? Answer True or False and explain why."
    elif eval_mode in ['multiple-choice', 'multiple-choice-none-option']:
        question = f"{DEFAULT_IMAGE_TOKEN} This is the target image.\n"
        options = ["B", "C", "D"][:num_choices]
        for idx, option in enumerate(options):
            question += f"{DEFAULT_IMAGE_TOKEN} This is image {option}.\n"
        question += f"In image {' , '.join(options)}, which one is the same person as the target image (answer E if none) and explain why."
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
    return question


def parse_response(response_text, eval_mode):
    response_text = response_text.strip().lower()
    if eval_mode == 'pairs':
        if 'true' in response_text:
            return 'True'
        elif 'false' in response_text:
            return 'False'
        elif 'yes' in response_text:
            return 'True'
        elif 'no' in response_text:
            return 'False'
        else:
            return 'Unknown'
    elif eval_mode in ['multiple-choice', 'multiple-choice-none-option']:
        for char in response_text.upper():
            if char in ['B', 'C', 'D', 'E']:
                return char
        # Default to 'E' if none of the options are found
        return 'E'
    else:
        return 'Unknown'


def compute_accuracy(results, eval_mode):
    if eval_mode == 'pairs':
        correct = sum(
            1 for res in results if res["model_answer"] == res["correct_option"]
        )
    elif eval_mode in ['multiple-choice', 'multiple-choice-none-option']:
        correct = sum(
            1 for res in results if res["model_answer"].strip() == res["correct_option"].strip()
        )
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
    
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0.0
    return accuracy


class Evaluator:
    def __init__(self, model, tokenizer, image_processor, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.args = args

    def evaluate(self, data_samples):
        results = []
        for sample in tqdm(data_samples[:self.args.num_samples], desc="Evaluating"):
            person_name = sample['person_name']
            target_image_path = sample['target_image']
            choices_paths = sample['choices']
            correct_option = sample['correct_option']
            labels = sample['labels']

            # Load images
            target_image = Image.open(target_image_path)
            choices_images = [Image.open(img_path) for img_path in choices_paths]
            images = [target_image] + choices_images  # Images: target_image, choices

            # Generate prompt
            question = generate_prompt(self.args.eval_mode, len(choices_images))

            # Process images
            image_tensors = process_images(images, self.image_processor, self.model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
            image_sizes = [image.size for image in images]

            # Prepare interleaved text-image input
            conv_template = "qwen_1_5"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)

            # Generate response
            cont = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            response = text_outputs[0]

            # Parse the response to get the answer
            model_answer = parse_response(response, self.args.eval_mode)

            # Record the result
            result = {
                "person_name": person_name,
                "correct_option": correct_option,
                "model_answer": model_answer,
                "question": question,
                "response": response,
                "images": {
                    "target": os.path.basename(target_image_path),
                }
            }
            for idx, choice_path in enumerate(choices_paths):
                choice_label = chr(ord('B') + idx)
                result["images"][choice_label] = os.path.basename(choice_path)
            results.append(result)

        return results


def main():
    # Set the process name
    setproctitle.setproctitle("LLaVA-Face")

    # Set random seed for reproducibility
    random.seed(42)

    args = parse_arguments()

    # Load model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained = os.path.join(base_dir, f"../../../../models/llava-onevision-qwen2-{args.model_size}-ov")
    model_name = "llava_qwen"
    device = "cuda"
    llava_model_args = {"multimodal": True}
    overwrite_config = {"image_aspect_ratio": "pad"}
    llava_model_args["overwrite_config"] = overwrite_config
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device, **llava_model_args
    )
    model.eval()

    # Load dataset
    lfw_dir = os.path.join(base_dir, "./dataset/lfw_funneled")
    if args.eval_dataset == 'lfw':
        person_folders, all_images = load_lfw_dataset(lfw_dir)

    # Setup results file
    results_file, existing_results = setup_results_file(base_dir, args)

    # Create data samples
    if args.eval_mode == 'pairs':
        data_samples = create_pairs_data(person_folders, all_images)
    elif args.eval_mode in ['multiple-choice', 'multiple-choice-none-option']:
        data_samples = create_multiple_choice_data(person_folders, all_images, args.eval_mode)
    else:
        raise ValueError(f"Unknown evaluation mode: {args.eval_mode}")

    # Evaluate
    evaluator = Evaluator(model, tokenizer, image_processor, device, args)
    results = evaluator.evaluate(data_samples)

    # Compute and print accuracy
    accuracy = compute_accuracy(results, args.eval_mode)
    print(f"Accuracy: {accuracy:.2f}%")

    # Save results with final result at the beginning
    final_result = {
        "accuracy": accuracy
    }
    with open(results_file, 'w') as f:
        json.dump([final_result] + existing_results + results, f, indent=4)


if __name__ == "__main__":
    main()
