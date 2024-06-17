# DATA FILES

- `keypoints.txt'
  The original keypoints that summaried by LLM
  
- `indicators.txt'
  Our bias indicators.
  Format: ID, Indicator, Category, Lable, Confidence score
  
- `flipbias_testset.txt'
  The selected triples, each triple with Left, Center and Right articles in the same event, of dataset FlipBias that used in our experiments.
  
- `adp_topic_results.json'
  The bias predication results (number of instances in each 'Label-Prediction') of dataset ADP that presented by topics.
  Format: "Topic": \[Left-Left, Left-Center, Left-Right, Left-Unknown, Center-Left, Center-Center, Center-Right, Center-Unknown, Right-Left, Right-Center, Right-Right, Right-Unknown\]
  
- `flipbias_cluster_results.json'
  The bias predication results (number of instances in each 'Label-Prediction') of dataset FlipBias that presented by lantent topics.
  Format: "ID of Lantent Topic": \[Left-Left, Left-Center, Left-Right, Left-Unknown, Center-Left, Center-Center, Center-Right, Center-Unknown, Right-Left, Right-Center, Right-Right, Right-Unknown\]
  
