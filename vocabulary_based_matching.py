import os
import json
import argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string





# Initialize stop words
stopwords_set = set(stopwords.words('english') + list(string.punctuation))

# Mapping original labels to readable labels
align_label = {'left': 'Left', 'center': 'Center', 'right': 'Right'}

# Initialize frequency dictionaries
fre_total = {}
fre_right = {}
fre_center = {}
fre_left = {}

# Function to process articles and calculate word frequencies
def process_articles(data_dir):
    files = os.listdir(data_dir)
    for file in files:
        if file == '.DS_Store':
            continue
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r') as fcc_file:
            content_dict = json.load(fcc_file)
            process_article(content_dict)

def process_article(content_dict):
    content = content_dict['content'].strip()
    title = content_dict['title'].strip()
    ori_label = content_dict['bias_text']
    label = align_label[ori_label]
    all_content = title + content
    all_content_tokens = word_tokenize(all_content)

    for token in all_content_tokens:
        token = token.lower()
        if len(token) <= 2 or token in stopwords_set:
            continue
        update_frequency(token, label)

def update_frequency(token, label):
    fre_total[token] = fre_total.get(token, 0) + 1
    if label == 'Right':
        fre_right[token] = fre_right.get(token, 0) + 1
    elif label == 'Left':
        fre_left[token] = fre_left.get(token, 0) + 1
    elif label == 'Center':
        fre_center[token] = fre_center.get(token, 0) + 1

# Save frequency data to JSON
def save_frequency_data(filepath):
    fre_words_adp = {
        'left': fre_left,
        'center': fre_center,
        'right': fre_right,
        'total': fre_total
    }
    with open(filepath, "w") as f:
        json.dump(fre_words_adp, f)

# Calculate total tokens per label
def calculate_total_tokens(data_dir):
    left_totaltoken = 0
    right_totaltoken = 0
    center_totaltoken = 0

    files = os.listdir(data_dir)
    for file in files:
        if file == '.DS_Store':
            continue
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r') as fcc_file:
            content_dict = json.load(fcc_file)
            left_totaltoken, right_totaltoken, center_totaltoken = update_token_count(content_dict, left_totaltoken, right_totaltoken, center_totaltoken)

    print(left_totaltoken, center_totaltoken, right_totaltoken)
    return left_totaltoken, right_totaltoken, center_totaltoken

def update_token_count(content_dict, left_totaltoken, right_totaltoken, center_totaltoken):
    content = content_dict['content'].strip()
    title = content_dict['title'].strip()
    ori_label = content_dict['bias_text']
    label = align_label[ori_label]
    all_content = title + content
    len_tokens = len(word_tokenize(all_content))

    if label == 'Left':
        left_totaltoken += len_tokens
    elif label == 'Right':
        right_totaltoken += len_tokens
    elif label == 'Center':
        center_totaltoken += len_tokens
    
    return left_totaltoken, right_totaltoken, center_totaltoken

# Load frequency data from JSON
def load_frequency_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Calculate normalized frequencies
def calculate_normalized_frequencies(frequency_data, total_tokens):
    normalized_frequencies = {}
    for token, freq in frequency_data.items():
        normalized_frequencies[token] = freq / total_tokens
    return normalized_frequencies

# Select significant tokens
def select_significant_tokens(dict_left, dict_right, threshold=2):
    selected_left = {token: freq for token, freq in dict_left.items() if token not in dict_right or dict_left[token]/dict_right[token] >= threshold}
    selected_right = {token: freq for token, freq in dict_right.items() if token not in dict_left or dict_right[token]/dict_left[token] >= threshold}
    print(len(selected_left))
    print(selected_left)
    print(len(selected_right))
    print(selected_right)
    return selected_left, selected_right

# Process and classify results
def process_results(flipbias_gen_filepath, left_dict_list, right_dict_list):
    with open(flipbias_gen_filepath, 'r', encoding='utf-8') as f_res:
        all_lines = f_res.readlines()
    
    results = []
    for l_i in range(len(all_lines)):
        if '----------------------' == all_lines[l_i].strip():
            triple_id, true, pred, count_left, count_right = classify_line(all_lines, l_i, left_dict_list, right_dict_list)
            results.append((triple_id, true, pred, count_left, count_right))
    
    return results

def classify_line(all_lines, l_i, left_dict_list, right_dict_list):
    count_left = 0
    count_right = 0

    triple_id = all_lines[l_i + 1].split('\t')[0].strip()
    true = all_lines[l_i + 1].split('\t')[1].strip()
    answer = all_lines[l_i + 1].split('\t')[3].strip()

    all_answer_tokens = word_tokenize(answer)
    for answer_token in all_answer_tokens:
        answer_token_lower = answer_token.lower()
        if answer_token_lower in left_dict_list:
            count_left += 1
        if answer_token_lower in right_dict_list:
            count_right += 1

    # para_threshold is a parameter to define the threshold of bias detection
    para_threshold = 3
    if count_left > count_right + para_threshold:
        pred = 'Left'
    elif count_right > count_left + para_threshold:
        pred = 'Right'
    else:
        pred = 'Center'

    return triple_id, true, pred, count_left, count_right

# Main execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and analyze article biases")
    parser.add_argument("--data_dir", default='../data/Article-Bias-Prediction-main/data/jsons', type=str, help="Directory containing the article JSON files")
    parser.add_argument("--output_file", default="./fre_words_adp_lower.json", type=str, help="File to save the frequency data")
    parser.add_argument("--results_file", default='./flipbias_gen_320words_output.txt', type=str, help="File containing the results to be processed")

     
    args = parser.parse_args()

    data_dir = args.data_dir
    output_file = args.output_file
    results_file = args.results_file

    process_articles(data_dir)
    save_frequency_data(output_file)
    left_totaltoken, right_totaltoken, center_totaltoken = calculate_total_tokens(data_dir)

    frequency_data = load_frequency_data(output_file)
    fre_left = frequency_data['left']
    fre_right = frequency_data['right']
    fre_center = frequency_data['center']

    dict_left = calculate_normalized_frequencies(fre_left, left_totaltoken)
    dict_right = calculate_normalized_frequencies(fre_right, right_totaltoken)
    dict_center = calculate_normalized_frequencies(fre_center, center_totaltoken)

    selected_left, selected_right = select_significant_tokens(dict_left, dict_right)
    
    left_dict_list = list(selected_left.keys())
    right_dict_list = list(selected_right.keys())

    results = process_results(results_file, left_dict_list, right_dict_list)
    
    
    







