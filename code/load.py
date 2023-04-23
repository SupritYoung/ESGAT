import torch

PATH = "berttriplet-1.pt"
model_dict = torch.load(PATH).cpu()
print(model_dict)
