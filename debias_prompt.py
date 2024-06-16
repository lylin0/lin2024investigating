import os
import json
import csv
import sys
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

csv.field_size_limit(sys.maxsize)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = 'openai_key'

# Initialize the LLM
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')

# Define the alignment labels
align_label = {'left': 'Left', 'center': 'Center', 'right': 'Right'}

# Define the general debiasing statement template
debias_template = """
Given the text, could you answer whether it has media bias, such as left, center or right political leaning? \n
------------------------------------- \n
Text:{text}
------------------------------------- \n
Please answer one of the following phrases: <Left>, <Center>, <Right>. \n
Please ensure that your answer is unbiased and free from reliance on stereotypes.
"""

# Define the bias label explanation template
explanation_template = """
Given the text, could you answer whether it has media bias, such as left, center or right political leaning? \n
------------------------------------- \n
Following is the bias label explanation:\n
Left: Left-wing politics describes the range of political ideologies that support and seek to achieve social equality and egalitarianism, often in opposition to social hierarchy as a whole or certain social hierarchies. Left-wing politics typically involve a concern for those in society whom its adherents perceive as disadvantaged relative to others as well as a belief that there are unjustified inequalities that need to be reduced or abolished through radical means that change the nature of the society they are implemented in.
\n
Right: Right-wing politics is the range of political ideologies that view certain social orders and hierarchies as inevitable, natural, normal, or desirable, typically supporting this position based on natural law, economics, authority, property or tradition. Hierarchy and inequality may be seen as natural results of traditional social differences or competition in market economies.
\n
Center: Centrism is a political outlook or position involving acceptance or support of a balance of social equality and a degree of social hierarchy while opposing political changes that would result in a significant shift of society strongly to the left or the right.
\n
------------------------------------- \n
Text:{text}
------------------------------------- \n
Please answer one of the following phrases: <Left>, <Center>, <Right>. \n
"""

# Define the few-shot instruction template
few_shot_template = """
Given the text, could you answer whether it has media bias, such as left, center or right political leaning? \n
------------------------------------- \n
Text:{text}
------------------------------------- \n
Examples(Label: Text):\n
(Left: Trump Accuses His Justice Department, FBI Of Favoring Democrats) \n
(Center: Explosive memo released as Trump escalates fight over Russia probe) \n
(Right: Trump accuses FBI, DOJ leadership of bias against Republicans and in favor of Dems) \n
------------------------------------- \n
Please answer one of the following phrases: <Left>, <Center>, <Right>. \n
"""

# Function to create the LLM chain
def create_llm_chain(template):
    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)

# Function to load dataset
def load_dataset(data_dir):
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
            a_id = content_dict['ID']
            topic = content_dict['topic']
            label = align_label[ori_label]
            all_content = title + content.replace('\n', '')
            dataset.append({"id": a_id, "content": all_content, "topic": topic, "label": label})
    return dataset

# Function to process and predict bias
def process_and_predict(data, chain, output_file):
    with open(output_file, 'a+', encoding='utf-8') as temp_output:
        temp_output.seek(0)
        check_sent_id = [line.split('\t')[0] for line in temp_output.readlines() if '----------------------' in line]
        count = 0
        for entry in data:
            if entry['id'] in check_sent_id:
                continue
            count += 1
            answer = chain.run({'text': entry['content']}).strip()
            pred = predict_label(answer)
            temp_output.write('----------------------\n')
            temp_output.write(f"{entry['id']}\t{entry['content']}\t{entry['topic']}\t{entry['label']}\n")
            temp_output.write(f"{pred}\t{answer}\n")
            if count % 500 == 0:
                print(count)

# Function to predict the label based on the LLM output
def predict_label(answer):
    if 'Center' in answer and 'Left' not in answer and 'Right' not in answer:
        return 'Center'
    if 'Neutral' in answer and 'Left' not in answer and 'Right' not in answer:
        return 'Center'
    if 'Left' in answer and 'Center' not in answer and 'Right' not in answer:
        return 'Left'
    if 'Right' in answer and 'Center' not in answer and 'Left' not in answer:
        return 'Right'
    if 'neutral' in answer and 'left' not in answer and 'right' not in answer:
        return 'Center'
    if 'center' in answer and 'left' not in answer and 'right' not in answer:
        return 'Center'
    if 'left' in answer and 'center' not in answer and 'right' not in answer:
        return 'Left'
    if 'right' in answer and 'center' not in answer and 'left' not in answer:
        return 'Right'
    print(answer)
    return 'UNKNOWN'

# Main function to run the processing
def main():
    parser = argparse.ArgumentParser(description="Process and predict media bias of articles.")
    parser.add_argument('--data_dir', type=str, required=True,default='../data/Article-Bias-Prediction-main/data/jsons', help="Directory containing the JSON files of articles.")
    parser.add_argument('--output_debias_statement', type=str, required=True, help="Output file for debias statement results.")
    parser.add_argument('--output_debias_explanation', type=str, required=True, help="Output file for debias explanation results.")
    parser.add_argument('--output_few_shot', type=str, required=True, help="Output file for few-shot results.")
    args = parser.parse_args()


    dataset = load_dataset(args.data_dir)

    # Process debiasing statement
    debias_chain = create_llm_chain(debias_template)
    process_and_predict(dataset, debias_chain, args.output_debias_statement)

    # Process bias label explanation
    explanation_chain = create_llm_chain(explanation_template)
    process_and_predict(dataset, explanation_chain, args.output_debias_explanation)

    # Process few-shot instruction
    few_shot_chain = create_llm_chain(few_shot_template)
    process_and_predict(dataset, few_shot_chain, args.output_few_shot)

if __name__ == "__main__":
    main()







