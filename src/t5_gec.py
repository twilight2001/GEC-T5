from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset
import argparse
import tqdm
class TrainerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, sep = "\t", names=['original_sentence', 'edited_sentence'])
        df = df.dropna()
        df.reset_index(inplace=True)
        self.dataset= df

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source = self.dataset.loc[idx, 'original_sentence']
        target = self.dataset.loc[idx, 'edited_sentence']

        input_ids = self.tokenizer.encode(source, return_tensors='pt', padding='max_length',truncation='longest_first', max_length=64)[0]
        label = self.tokenizer.encode(target, return_tensors='pt', padding='max_length',truncation='longest_first', max_length=64)[0]

        return {'input_ids':input_ids, 'labels':label}


train_data_path = '/data/asadul/research/GEC-T5/data/wiki.tok.small.tsv'
output_path = '/data/asadul/research/GEC-T5/output/'
epoch = 3
batch_size=64
weight_decay = 5e-5
lr = 3e-4
gra_acc_steps=6
save_steps=100
model = T5ForConditionalGeneration.from_pretrained('t5-small')

model.to('cuda')
training_args = TrainingArguments(
    output_dir=output_path,
    num_train_epochs=epoch,
    logging_dir='/data/asadul/research/GEC-T5/logs',
    save_steps=100,
)
train_dataset = TrainerDataset(train_data_path)
print(train_dataset)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
model = T5ForConditionalGeneration.from_pretrained(output_path+'checkpoint-600/')
def predict(tokenizer, model, passages):
    inputs = tokenizer(passages, return_tensors="pt", padding='max_length',truncation='longest_first', max_length=64)
    queries = model.generate(inputs['input_ids'], num_beams=12 ,max_length=160)
    queries_decode = []
    for query in queries:
        query_decode = tokenizer.decode(query, skip_special_tokens=True)
        queries_decode.append(query_decode)
    return queries_decode

tokenizer = T5Tokenizer.from_pretrained('t5-small')
test_data = '../data/nucle.src'
output = 'nucle.pred'
pred_batch_size=1
with open(test_data) as f,open(output, 'w') as f_out:
    lines = f.read().splitlines()
    for index in range(0, len(lines), pred_batch_size):
        batch = lines[index:index+pred_batch_size]
        sys = predict(train_dataset.tokenizer, model, batch)
        print(batch[0])
        print(sys[0])
        for output in sys:
            f_out.write(output+'\n')
