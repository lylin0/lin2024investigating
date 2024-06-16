import os
import re
import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and predict media bias of articles by different llms.")
    parser.add_argument('--model_name', type=str, required=True, choices=[
        "meta-llama/Llama-2-7b-chat-hf",
        "daryl149/llama-2-7b-chat-hf",
        "lmsys/vicuna-7b-v1.5",
        "mistralai/Mistral-7B-v0.1"
    ], help="Choose different LLMs.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--output_file', type=str, required=True, help="Output file path.")
    return parser.parse_args()

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    return tokenizer, pipeline

def predict_bias(text, pipeline, tokenizer):
    input_prompt = "Given the text, could you answer whether it has media bias, such as left, center or right political leaning? \n"
    input_prompt2 = "Please answer one of the following phrases: <Left>, <Center>, <Right>.\n Answer:"
    try:
        sequences = pipeline(
            input_prompt + text + input_prompt2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=2048,
        )
        ori_answer = sequences[0]['generated_text']
        answer = re.findall('Answer:(.*)', ori_answer)[0].strip()
    except Exception as e:
        print(f"Error during prediction: {e}")
        answer = "UNKNOWN"
    return answer

def determine_label(answer):
    if 'Center' in answer and 'Left' not in answer and 'Right' not in answer:
        return 'Center'
    elif 'Neutral' in answer and 'Left' not in answer and 'Right' not in answer:  
        return 'Center'
    elif 'Left' in answer and 'Center' not in answer and 'Right' not in answer:
        return 'Left'
    elif 'Right' in answer and 'Center' not in answer and 'Left' not in answer:
        return 'Right'
    elif 'neutral' in answer and 'left' not in answer and 'right' not in answer:
        return 'Center'
    elif 'center' in answer and 'left' not in answer and 'right' not in answer:
        return 'Center'
    elif 'left' in answer and 'center' not in answer and 'right' not in answer:
        return 'Left'
    elif 'right' in answer and 'center' not in answer and 'left' not in answer:
        return 'Right'
    else:
        print(f"Unclear answer: {answer}")
        return 'UNKNOWN'

def main():
    args = parse_arguments()

    tokenizer, pipeline = load_model(args.model_name)
    data_path = args.data_path
    output_file_path = args.output_file

    with open(data_path, 'r') as data_file, open(output_file_path, 'a+', encoding='utf-8') as output_file:
        output_file.seek(0)
        processed_ids = {line.split('\t')[0] for line in output_file.readlines() if '----------------------' in line}
        
        count = 0
        for line in data_file:
            count += 1
            items = line.strip().split('\t')
            fb_id, _, text, label = items
            if fb_id in processed_ids:
                continue

            answer = predict_bias(text, pipeline, tokenizer)
            pred_label = determine_label(answer)
            
            output_file.write('----------------------\n')
            output_file.write(f"{fb_id}\t{items[1]}\t{text}\t{label}\n")
            output_file.write(f"{answer}\n")
            output_file.write(f"{pred_label}\n")

            if count % 500 == 0:
                print(f"Processed {count} entries.")

if __name__ == "__main__":
    main()







