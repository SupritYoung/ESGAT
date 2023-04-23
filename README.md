# ES-GAT

The code of our paper "Improving Aspect Sentiment Triplet Extraction with Perturbed Masking and Edge-Enhanced Sentiment Graph Attention Network" accepted by IJCNN 2023.
## Requirements

- python==3.7.6

- torch==1.4.0
- transformers==3.4.0
- argparse==1.1

## Training

To train the model, run:

```
cd ./code
sh run.sh
```
or
```
python main.py --mode train --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 1000 --pooling avg --prefix ../data/D2/
```

## Acknowledge


The basic code framework is based on ["https://github.com/CCChenhao997/EMCGCN-ASTE"](https://github.com/CCChenhao997/EMCGCN-ASTE), thanks for the contribution of its open source code.