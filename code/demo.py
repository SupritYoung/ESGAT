import torch
import pickle
from prepare_vocab import VocabHelp

PATH = "savemodel/berttriplet.pt"
# GPU 模型 转 CPU
model = torch.load(PATH, map_location=lambda storage, loc: storage)
# postag 的嵌入
print(model)
