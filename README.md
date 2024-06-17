# Investigating Bias in LLM-Based Bias Detection: Disparities between LLMs and Human Perception
This repository contains the code for the experiments presented in the paper "Investigating Bias in LLM-Based Bias Detection: Disparities between LLMs and Human Perception". 

# Experiment Details
- \`bias_prediction.py', `bias_prediction_otherllms.py'
  Prompt LLMs to detect the bias and LLMs should label Each article as 'Left', 'Center', or 'Right'. 

- `continuous_writing.py'
  Generates articles based on provided prefixes and evaluates the generated content for bias.

- \`debias_finetune.py', `debias_prompt.py'
  Debias experiments on fine-tuning the model and prompts to isolate bias.
  
- `topic_construction.py'
  Construction of latent topics in bias indicators.
  
- `vocabulary_based_matching.py'
  Construction of a vocabulary dictionary used to measure media bias based on vocabulary.
