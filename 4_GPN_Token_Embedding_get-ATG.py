import gpn.model
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch,sys
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, sequences, names, tokenizer):
        self.sequences = sequences
        self.names = names
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        name = self.names[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
        return {
            'sequence': sequence,
            'name': name,
            'input_ids': encoding['input_ids']
        }

device = sys.argv[3]
model_path = 'songlab/gpn-brassicales'

model = AutoModel.from_pretrained(model_path)
model.to(device)
model.eval()


# Initialize your tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

sequences = []
names = []
for record in SeqIO.parse(sys.argv[1], "fasta"):
    sequences.append(str(record.seq))
    names.append(record.id)

# Create dataset
dataset = SequenceDataset(
    sequences=sequences,
    tokenizer=tokenizer,
    names=names
)
# Create your data loader
loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)
print("Total batches: ", len(loader))

for batch in loader:
    curName = np.array(batch['name'])[:,np.newaxis]
    curIDs = batch['input_ids'].to(device)
    curIDs = curIDs.squeeze(1)
    with torch.inference_mode():
        embedding = model(input_ids=curIDs).last_hidden_state
    token_embeddings = [StandardScaler().fit_transform(curEmd.cpu().numpy()) for curEmd in embedding]
    res = np.stack(token_embeddings)
    center_2_embeddings = np.mean(res[:,255:257,:], axis = 1)
    final_embeddings = np.concatenate((curName, center_2_embeddings), axis = 1)
    with open(sys.argv[2], 'a') as f:
        np.savetxt(f, final_embeddings, delimiter = '\t', fmt='%s')
