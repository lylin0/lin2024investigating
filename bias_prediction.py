import os
import json
import csv
import sys
import argparse
from sklearn.metrics import precision_recall_fscore_support
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Set the maximum field size limit for CSV
csv.field_size_limit(sys.maxsize)

# Define constants and initialize LLM
os.environ["OPENAI_API_KEY"] = 'openaikey'
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
# llm = ChatOpenAI(model_name='gpt-4-turbo') 

# Define the template and chain
gen_template = """
Given the text, could you answer whether it has media bias, such as left, center or right political leaning? \n
------------------------------------- \n
Text:{text}
------------------------------------- \n
Please answer one of the following phrases: <Left>, <Center>, <Right>
"""
prompt = ChatPromptTemplate.from_template(gen_template)
chain = LLMChain(llm=llm, prompt=prompt)

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Media Bias Classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    return parser.parse_args()

# Load dataset
def load_dataset(data_path):
    dataset = {}
    with open(data_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[0] not in dataset:
                dataset[row[0]] = {}
            dataset[row[0]][row[1]] = row[5] + row[6]
    return dataset

# Create test set
def create_test_set(dataset, label2id, testset_path):
    count_all = count_test = 0
    with open(testset_path, 'a', encoding='utf-8') as f_testset:
        for k in dataset.keys():
            count_all += len(dataset[k].keys())
            if 'From the Left' in dataset[k] and 'From the Right' in dataset[k] and 'From the Center' in dataset[k]:
                for s_k in dataset[k].keys():
                    count_test += 1
                    text = dataset[k][s_k]
                    label = label2id[s_k]
                    f_testset.write(f"{count_test}\t{k}\t{text}\t{label}\n")
    print(count_test, count_all)

# Check already processed IDs
def check_processed_ids(output_path):
    with open(output_path, 'a+', encoding='utf-8') as temp_output:
        temp_output.seek(0)
        check_sent_id = [line.split('\t')[0] for line in temp_output if '----------------------' in line]
    return check_sent_id

# Predict biases
def predict_biases(data_path, check_sent_id, output_path, chain):
    preds, trues = [], []
    count = 0

    with open(data_path) as f_fb, open(output_path, 'a+', encoding='utf-8') as temp_output:
        all_datas = f_fb.readlines()

        for line in all_datas:
            count += 1
            items = line.split('\t')
            fb_id = items[0]
            if fb_id in check_sent_id:
                continue
            text = items[2]
            label = items[3].strip()
            trues.append(label)

            # Get GPT prediction
            answer = chain.run({'text': text}).strip()

            if 'Center' in answer and 'Left' not in answer and 'Right' not in answer:
                pred = 'Center'
            elif 'Left' in answer and 'Center' not in answer and 'Right' not in answer:
                pred = 'Left'
            elif 'Right' in answer and 'Center' not in answer and 'Left' not in answer:
                pred = 'Right'
            else:
                pred = 'UNKNOWN'
                print(answer)

            preds.append(pred)
            temp_output.write(f'----------------------\n{fb_id}\t{items[1]}\t{text}\t{label}\n')
            temp_output.write(f'{pred}\t{answer}\n')

            if count % 500 == 0:
                print(count)

# Evaluate results
def evaluate_results(output_path):
    with open(output_path, 'r', encoding='utf-8') as f_res:
        all_lines = f_res.readlines()

    results = {
        'Left': {'Left': 0, 'Center': 0, 'Right': 0, 'UNKNOWN': 0},
        'Center': {'Left': 0, 'Center': 0, 'Right': 0, 'UNKNOWN': 0},
        'Right': {'Left': 0, 'Center': 0, 'Right': 0, 'UNKNOWN': 0}
    }

    for i in range(len(all_lines)):
        if '----------------------' in all_lines[i]:
            true = all_lines[i + 1].split('\t')[3].strip()
            pred = all_lines[i + 2].split('\t')[0].strip()
            if true in results:
                if pred in results[true]:
                    results[true][pred] += 1

    for true_label, counts in results.items():
        print(f"{true_label}: {counts}")

def main():
    args = parse_args()

    # Label mapping
    label2id = {"From the Right": 'Right', "From the Left": 'Left', "From the Center": 'Center'}

    # Load dataset and create test set
    dataset = load_dataset(args.data_path)
    create_test_set(dataset, label2id, './flipbias_testset.txt')

    # Check processed IDs
    check_sent_id = check_processed_ids(args.output_path)

    # Predict biases
    predict_biases('./flipbias_testset.txt', check_sent_id, args.output_path, chain)

    # Evaluate results
    evaluate_results(args.output_path)

if __name__ == "__main__":
    main()















