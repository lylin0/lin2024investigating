
import os
import json
import random
import argparse
import openai
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Bias Prediction Model Training and Evaluation")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing JSON data files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files")
    parser.add_argument("--test_file", type=str, required=True, help="Test file path")
    parser.add_argument("--model_id", type=str, required=True, help="Fine-tuned model ID")
    return parser.parse_args()

# Function to load dataset
def load_dataset(data_dir, label_filter=None):
    align_label = {'left': 'Left', 'center': 'Center', 'right': 'Right'}
    dataset = []
    files = os.listdir(data_dir)
    for file in files:
        if file == '.DS_Store':
            continue
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r') as fcc_file:
            content_dict = json.load(fcc_file)
            content = content_dict['content'].strip()
            title = content_dict['title'].strip()
            ori_label = content_dict['bias_text']
            label = align_label[ori_label]
            all_content = title + content
            each_article = {"text": all_content.replace('\n', ''), "label": label}
            if label_filter is None or label == label_filter:
                dataset.append(each_article)
    return dataset

# Function to save dataset to JSON file
def save_dataset(dataset, output_file):
    data = {'data': dataset}
    with open(output_file, "w") as f:
        json.dump(data, f)

# Construct datasets
def construct_datasets(data_dir, output_dir):
    dataset_left = load_dataset(data_dir, 'Left')
    dataset_right = load_dataset(data_dir, 'Right')
    dataset_center = load_dataset(data_dir, 'Center')

    select_left = random.sample(dataset_left, 300)
    save_dataset(select_left, os.path.join(output_dir, "adp_rd300left.json"))

    select_right = random.sample(dataset_right, 300)
    save_dataset(select_right, os.path.join(output_dir, "adp_rd300right.json"))

    left150 = random.sample(select_left, 150)
    select_center = random.sample(dataset_center, 150)
    adp_rd150left150center = left150 + select_center
    save_dataset(adp_rd150left150center, os.path.join(output_dir, "adp_rd150left150center.json"))

    dataset_all = load_dataset(data_dir)
    select_data = random.sample(dataset_all, 300)
    save_dataset(select_data, os.path.join(output_dir, "adp_rd300.json"))

# Transform JSON to JSONL format
def transform_jsonl(input_file_path, output_file_path):
    system_content = "You are an assistant that helps predict the political leaning of the input article. Reply with the media bias with Right, Center, or Left."
    entries = []
    with open(input_file_path, 'r') as file:
        data = json.load(file)
        train_set = data['data']
        for instance in train_set:
            entry = {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": instance["text"]},
                    {"role": "assistant", "content": instance["label"]}
                ]
            }
            entries.append(entry)
    with open(output_file_path, 'w') as outfile:
        for entry in entries:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

# Submit file to OpenAI and create fine-tuning job
def submit_and_fine_tune(client, file_path, training_file_id):
    print(client.files.create(file=open(file_path, "rb"), purpose="fine-tune"))
    print(client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo",
        hyperparameters={"n_epochs": 3, "batch_size": 32, "learning_rate_multiplier": 2}
    ))

# Evaluate the model
def evaluate_model(client, test_file_path, model_id):
    system_content = "You are an assistant that helps predict the political leaning of the input article. Reply with the media bias with Right, Center, or Left."
    with open(test_file_path) as f:
        all_datas = f.readlines()

    preds = []
    trues = []
    for line in all_datas:
        items = line.split('\t')
        text = items[2]
        label = items[3].strip()

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": text[:15000]}]

        try:
            response = client.chat.completions.create(model=model_id, messages=messages)
            pred = response.choices[0].message.content.strip()
            trues.append(label)
            preds.append(pred)
            print(label, pred)
        except Exception as e:
            print(f"Error: {e}")
            with open('results_wrong.json', 'w') as f:
                json.dump({"ground_truth": trues, "predict_label": preds}, f)

    with open('results_flipbias_ftrd300left.json', 'w') as f:
        json.dump({"ground_truth": trues, "predict_label": preds}, f)

# Main script execution
if __name__ == "__main__":
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.api_key
    client = OpenAI()

    # Construct datasets
    construct_datasets(args.data_dir, args.output_dir)

    # Transform datasets to JSONL format
    for dataset in ['adp_rd300left', 'adp_rd150left150center', 'adp_rd300', 'adp_rd300right']:
        transform_jsonl(os.path.join(args.output_dir, f"{dataset}.json"), os.path.join(args.output_dir, f"{dataset}.jsonl"))

    # Submit files to OpenAI and create fine-tuning jobs
    # Replace "file-AR1njoz6jd8fAtkxneNZk8NW" with your actual file ID after submission
    for file in ['adp_rd300left.jsonl', 'adp_rd150left150center.jsonl', 'adp_rd300.jsonl', 'adp_rd300right.jsonl']:
        submit_and_fine_tune(client, os.path.join(args.output_dir, file), "file-AR1njoz6jd8fAtkxneNZk8NW")

    # Evaluate the fine-tuned model on the test set
    evaluate_model(client, args.test_file, args.model_id)










