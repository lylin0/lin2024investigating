
import os
import csv
import json
import sys
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Define the prompt template
gen_template = """
According to the given text (Title and the beginning of a piece of news), write the whole article. \n
------------------------------------- \n
Text:{text}
"""
prompt = ChatPromptTemplate.from_template(gen_template)

# Function to select triple IDs
def load_tri_ids(file_path):
    tri_ids = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            t_id = line.strip().split('\t')[1]
            if t_id not in tri_ids:
                tri_ids.append(t_id)
    return tri_ids

# Function to load titles and contents
def load_titles_and_contents(file_path, check_tri_id, label2id):
    titleset = {}
    contentset = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[0] not in check_tri_id:
                continue
            bias_label = label2id[row[1]]
            title = row[2]
            ori_title = row[5].replace('\n', '')
            ori_content = row[6].replace('\n', '')
            if row[0] not in titleset.keys():
                titleset[row[0]] = {}
                contentset[row[0]] = {}
            titleset[row[0]][bias_label] = ori_title
            contentset[row[0]][bias_label] = ori_content
    return titleset, contentset

# Function to save dataset embeddings
def save_embeddings(file_path, check_tri_id, label2id, prefix_length=0):
    dataset = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[0] not in check_tri_id:
                continue
            bias_label = label2id[row[1]]
            if row[0] not in dataset.keys():
                dataset[row[0]] = {}
            text = ' '.join(word_tokenize(row[6])[prefix_length:])
            dataset[row[0]][bias_label] = {'text': text, 'embedding': OpenAIEmbeddings().embed_query(text)}
    return dataset

# Function to generate articles and calculate similarity
def generate_and_evaluate_articles(titleset, contentset, dataset, dataset_wo, output_path):
    res_wo20words_embedding = {}
    count = 0

    with open(output_path, 'a', encoding='utf-8') as op:
        for tri_id in titleset.keys():
            if tri_id in check_tri_id:
                for title_label in titleset[tri_id].keys():
                    title = titleset[tri_id][title_label]
                    content = contentset[tri_id][title_label]
                    title_with_prefix = title + ' '.join(word_tokenize(content)[:20])

                    # Generate article with GPT
                    answer = chain.run({'text': title_with_prefix}).replace('\n', ' ').replace('\t', ' ')

                    # Calculate embeddings
                    embedding_answer = OpenAIEmbeddings().embed_query(answer)
                    embedding_left = dataset[tri_id]['Left']['embedding']
                    embedding_center = dataset[tri_id]['Center']['embedding']
                    embedding_right = dataset[tri_id]['Right']['embedding']
                    embedding_left_wo = dataset_wo[tri_id]['Left']['embedding']
                    embedding_center_wo = dataset_wo[tri_id]['Center']['embedding']
                    embedding_right_wo = dataset_wo[tri_id]['Right']['embedding']

                    if tri_id not in res_wo20words_embedding.keys():
                        res_wo20words_embedding[tri_id] = {}
                    res_wo20words_embedding[title_label] = embedding_answer

                    # Calculate similarity scores
                    scores = cosine_similarity([embedding_answer], [embedding_left, embedding_center, embedding_right])[0]
                    pred = ['Left', 'Center', 'Right'][scores.argmax()]
                    if max(scores) == scores.mean():
                        pred = 'UNKNOWN'
                        print(scores)

                    scores_wo = cosine_similarity([embedding_answer], [embedding_left_wo, embedding_center_wo, embedding_right_wo])[0]
                    pred_wo = ['Left', 'Center', 'Right'][scores_wo.argmax()]
                    if max(scores_wo) == scores_wo.mean():
                        pred_wo = 'UNKNOWN'
                        print(scores_wo)

                    # Write results to file
                    op.write('----------------------\n')
                    op.write(f"{tri_id}\t{title_label}\t{title_with_prefix}\t{answer}\n")
                    op.write(f"{pred}\t{' '.join(map(str, scores))}\n")
                    op.write(f"{pred_wo}\t{' '.join(map(str, scores_wo))}\n")

                    count += 1
                    if count % 500 == 0:
                        print(count)

    with open(res_embedding_path, "w") as f:
        json.dump(res_wo20words_embedding, f)

# Function to calculate result statistics
def calculate_statistics(file_path, prefix=False):
    with open(file_path, encoding='utf-8') as f_res:
        all_lines = f_res.readlines()

    stats = {
        'left_left': 0, 'left_center': 0, 'left_right': 0, 'left_unknown': 0,
        'center_left': 0, 'center_center': 0, 'center_right': 0, 'center_unknown': 0,
        'right_left': 0, 'right_center': 0, 'right_right': 0, 'right_unknown': 0
    }

    for l_i in range(len(all_lines)):
        if '----------------------' == all_lines[l_i].strip():
            true = all_lines[l_i + 1].split('\t')[1].strip()
            pred = all_lines[l_i + 2].split('\t')[0].strip() if not prefix else all_lines[l_i + 3].split('\t')[0].strip()

            key = f"{true.lower()}_{pred.lower()}"
            if key in stats:
                stats[key] += 1
            else:
                print(true, pred)
                print(all_lines[l_i + 1].split('\t')[0].strip())

    return stats

# Main function
def main(args):
    os.environ["OPENAI_API_KEY"] = args.api_key

    # Initialize OpenAI model
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
    chain = LLMChain(llm=llm, prompt=prompt)

    # Load triple IDs
    check_tri_id = load_tri_ids(args.flipbias_test_path)

    # Load titles and contents
    titleset, contentset =  (args.data_path, check_tri_id, label2id)

    # Save embeddings with and without prefix
    dataset = save_embeddings(args.data_path, check_tri_id, label2id, prefix_length=0)
    with open(args.flipbias_embedding_path, "w") as f:
        json.dump(dataset, f)

    dataset_wo = save_embeddings(args.data_path, check_tri_id, label2id, prefix_length=args.predix_length)
    with open(args.flipbias_embedding_wo_path, "w") as f:
        json.dump(dataset_wo, f)

    # Generate articles and evaluate
    generate_and_evaluate_articles(titleset, contentset, dataset, dataset_wo, args.output_path)

    # Calculate and print statistics for original and prefix-deleted results
    original_stats = calculate_statistics(args.output_path, prefix=False)
    print('original:', original_stats)

    prefix_stats = calculate_statistics(args.output_path, prefix=True)
    print('del prefix:', prefix_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue writing news articles.')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--data_path', type=str, default='./data_public.csv', help='Path to the CSV data file')
    parser.add_argument('--flipbias_test_path', type=str, default='./flipbias_testset.txt', help='Path to the flipbias test file')
    parser.add_argument('--predix_length', type=str, default='20', help='Length of the prefix giving LLM')
    parser.add_argument('--flipbias_embedding_path', type=str, default='./flipbias_article_embedding.json', help='Path to save embeddings')
    parser.add_argument('--flipbias_embedding_wo_path', type=str, default='./flipbias_article_embedding_wo20words.json', help='Path to save embeddings without prefix')
    parser.add_argument('--output_path', type=str, default='./flipbias_gen_20words_output.txt', help='Output path for generated articles')

    args = parser.parse_args()
    main(args)



    








