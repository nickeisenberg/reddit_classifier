from typing import cast
import torch
from torch.utils.data import Dataset
import os
from transformers import BertTokenizer


class TextFolderWithBertTokenizer(Dataset):
    def __init__(self, root_dir: str, which: str, instructions: dict | None = None, 
                 label_id_map: dict | None = None, max_length: int = 256):


        assert which in ["train", "val", "test"]

        self.root_dir = root_dir
        self.which = which
        self.instructions = instructions
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.txt_file_names, self.labels, self.label_to_id = self._load_files(
            self.root_dir, self.which, self.instructions
        )

        if label_id_map is not None:
            self.label_to_id = label_id_map

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}


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

        input_ids = cast(torch.Tensor, input["input_ids"])
        input_masks = cast(torch.Tensor, input["attention_mask"])

        label_id = self.label_to_id[self.labels[idx]]

        return torch.flatten(input_ids), torch.flatten(input_masks), torch.tensor(label_id)

        
    @staticmethod
    def _load_files(root_dir: str, which:str, instructions: dict | None = None):
        txt_file_names = []
        labels = []
        label_to_id = {}
        for label_name in os.listdir(root_dir):
            
            if instructions:
                if label_name in instructions:
                    if instructions[label_name] == 'ignore':
                        continue
                    else:
                        label_name, og_label_name = instructions[label_name], label_name
                else:
                    og_label_name = label_name
            else:
                og_label_name = label_name

            class_dir = os.path.join(root_dir, og_label_name, which)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith(".txt"):
                        labels.append(label_name)
                        txt_file_names.append(os.path.join(root, file))

        for name in labels:
            if name not in label_to_id:
                label_to_id[name] = len(label_to_id)

        return txt_file_names, labels, label_to_id


if __name__ == "__main__":
    dataset = TextFolderWithBertTokenizer(
        "data", 
        "train",
       {
            "movies": "ignore",
            "CryptoCurrency": "stocks",
            "wallstreetbets": "stocks",
            "formula1": "sports",
            "soccer": "sports"
       }
    )
    dataset.label_to_id
