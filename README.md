# CIFE: Causally Interpretable Driver Gene Prediction with Feature Representation

n this study, we present CIFE, a causally interpretable deep learning framework built upon multi-omics data and feature pre-training. The CIFE method incorporates easily interpretable multi-omics features, a feature pre-training module based on SCARF, and a CASTLE module that supports both classification and causal interpretation. By integrating multi-omics data (including mutations, copy number alterations, DNA methylation, and gene expression), CIFE achieves accurate identification of driver genes while generating a causal contribution adjacency matrix of features and predictions. This matrix, which depicts the relationship of features to features and features to prediction outcomes, captures the intrinsic mechanisms underlying predictions and refines the catalog of cancer driver genes. On the test set, CIFE demonstrates higher accuracy and identifies reliable candidate driver genes compared to state-of-the-art approaches. We discovered that gene connectivity centrality and SNP features within gene interaction networks exhibit the most substantial causal influence on driver gene predictions. Furthermore, CIFE revealed that the absence of promoters and terminators exerts a significant impact on gene expression. We are confident that CIFE can precisely pinpoint cancer driver genes while concurrently unraveling causal connections within the model’s decision-making process.

## Install

```
cd CIFE
pip install -r requirement.txt
```

## Usage

`cd codes`

for example:

train the model

```
python main_pytorch.py -t PANCAN -m train 
```

test the model

```
python main_pytorch.py -t PANCAN -m test 
```

get the casual interpretation of the model

```
python main_pytorch.py -t PANCAN -m casual 
```