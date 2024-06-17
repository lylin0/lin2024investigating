import os
import json
import re
import numpy as np
import argparse
from sklearn.cluster import AgglomerativeClustering
from langchain.embeddings.openai import OpenAIEmbeddings

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = 'openaikey'

# Function to get embeddings
def get_embeddings(input_file, output_file):
    with open(input_file, encoding='utf-8') as f_ip, open(output_file, "r+") as f:
        embedding_dict = json.load(f)
        count = 0
        for line in f_ip:
            count += 1
            sent = line.split('\t')[1].strip()
            if sent in embedding_dict:
                continue
            embedding_vector = OpenAIEmbeddings().embed_query(sent)
            embedding_dict[sent] = embedding_vector
            if count % 500 == 0:
                f.seek(0)
                json.dump(embedding_dict, f)
                f.truncate()
                print(count)
        f.seek(0)
        json.dump(embedding_dict, f)
        f.truncate()

# Function for strict clustering
def perform_clustering(embedding_file, input_file, output_file):
    with open(embedding_file, "r") as f:
        embedding_dict = json.load(f)
    
    with open(input_file, encoding='utf-8') as f_ip:
        embeds, texts = [], []
        for line in f_ip:
            sent = line.split('\t')[1].strip()
            sent_embedding = embedding_dict[sent]
            embeds.append(sent_embedding)
            texts.append(sent)

    X = np.array(embeds)
    clustering = AgglomerativeClustering(distance_threshold=2, n_clusters=None).fit(X)
    res_label = list(clustering.labels_)
    clu_lab = {texts[i]: str(res_label[i]) for i in range(len(texts))}

    with open(output_file, "w") as f:
        json.dump(clu_lab, f)

# Function to create indicator article dictionary
def create_indicator_article_dict(test_file, keypoints_file, output_file):
    article_selected2_dict = {}
    test2id_test1id = {}
    label2id = {'Left': '0', 'Center': '1', 'Right': '2'}
    
    with open(test_file, encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('\t')
            a_id, triple_id, label = items[0].strip(), items[1].strip(), items[3].strip()
            test1id = triple_id + label2id[label]
            test2id_test1id[a_id] = test1id
    
    category = ['Tone and Language', 'Sources and Citations', 'Coverage and Balance', 'Agenda and Framing', 'Examples and Analogies']
    temp_indicator_dict = {}

    with open(keypoints_file, encoding='utf-8') as f:
        for line in f:
            if '-----------------------' in line:
                items = line.split('\t')
                tri_id = items[0].strip()
                label = next(filter(lambda x: x in line, ['Left', 'Center', 'Right']), None)
                if label:
                    test1id = tri_id + label2id[label]
                    temp_indicator_dict[test1id] = []
            
            if match := re.findall(r'(.*? - .*? - .*?\n)', line):
                con = re.findall(r'(.*?) - (.*?) - (.*?)\n', line)[0]
                if con[0] not in category:
                    continue
                cat, ind = con[0].strip(), con[1].strip()
                pol = 'Left' if 'Left' in con[2] or 'left' in con[2] else 'Right' if 'Right' in con[2] or 'right' in con[2] else 'Center'
                temp_indicator_dict[test1id].append(ind)

    indicator_dict = {test2id: temp_indicator_dict[test2id_test1id[test2id]] for test2id in test2id_test1id}
    
    with open(output_file, "w") as f:
        json.dump(indicator_dict, f)

# Function to create cluster indicator dictionary
def create_cluster_indicator_dict(cluster_file, output_file):
    cluster_indicator_dict = {}
    with open(cluster_file, "r") as f:
        indicator_cluster = json.load(f)
    
    for indicator, cluster in indicator_cluster.items():
        if cluster not in cluster_indicator_dict:
            cluster_indicator_dict[cluster] = []
        cluster_indicator_dict[cluster].append(indicator)
    
    with open(output_file, "w") as f:
        json.dump(cluster_indicator_dict, f)

def main():
    parser = argparse.ArgumentParser(description='Process embeddings and clustering.')
    parser.add_argument('--embedding_input', type=str, default='./indicators.txt', help='Input file for embeddings')
    parser.add_argument('--embedding_output', type=str, default='./indicator_embedding.json', help='Output file for embeddings')
    parser.add_argument('--cluster_output', type=str, default='./indicator_cluster_dis_2.json', help='Output file for clustering results')
    parser.add_argument('--test_file', type=str, default='./flipbias_testset.txt', help='Test file for indicator article dictionary')
    parser.add_argument('--keypoints_file', type=str, default='./keypoints.txt', help='Keypoints file for indicator article dictionary')
    parser.add_argument('--indicator_article_output', type=str, default='./indicator_article_dict.json', help='Output file for indicator article dictionary')
    parser.add_argument('--cluster_file', type=str, default='./indicator_cluster_dis_2.json', help='Cluster file for creating cluster indicator dictionary')
    parser.add_argument('--cluster_indicator_output', type=str, default='./cluster_indicator_dict.json', help='Output file for cluster indicator dictionary')

    args = parser.parse_args()

    get_embeddings(args.embedding_input, args.embedding_output)
    perform_clustering(args.embedding_output, args.embedding_input, args.cluster_output)
    create_indicator_article_dict(args.test_file, args.keypoints_file, args.indicator_article_output)
    create_cluster_indicator_dict(args.cluster_file, args.cluster_indicator_output)

if __name__ == "__main__":
    main()


















