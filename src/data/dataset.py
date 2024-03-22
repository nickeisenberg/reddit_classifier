import torch
from torch.utils.data import Dataset
import os
from transformers import BertTokenizer


class TextFolderWithBertTokenizer(Dataset):
    def __init__(self, root_dir, max_length=256):
        self.root_dir = root_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.txt_file_names, self.labels, self.label_ids = self._load_files(
            self.root_dir
        )


    def __len__(self):
        return len(self.txt_file_names)


    def __getitem__(self, idx):
        file_path = self.txt_file_names[idx]
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.readline().strip()

        input = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        label = self.label_ids[self.labels[idx]]

        return input["input_ids"].reshape(-1), input["attention_mask"].reshape(-1), label

        
    @staticmethod
    def _load_files(root_dir):
        txt_file_names = []
        labels = []
        label_ids = {}
        for label_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label_name)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith(".txt"):
                        labels.append(label_name)
                        txt_file_names.append(os.path.join(root, file))

        for name in labels:
            if name not in label_ids:
                label_ids[name] = torch.tensor(len(label_ids))

        return txt_file_names, labels, label_ids
