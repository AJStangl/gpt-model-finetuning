import gc

import pandas
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

from src.tensor_encoding.tensor_encoding import TensorHelper


class RedditDataset(Dataset):
	"""
	TODO: Description of class
	"""
	def __init__(self, txt_list, tokenizer, max_length):
		self.input_ids = []
		self.attn_masks = []
		self.labels = []
		for txt in txt_list:
			encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
			self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
			self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.attn_masks[idx]

	# TODO: Pass Configure training arguments...
	def fine_tunining_new_model(self) -> None:
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

		tokenizer.save_pretrained('./results')

		model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

		df = pandas.read_csv("training.csv")

		conversations = df['TrainingString']

		valid_prompts = []
		for conversation in conversations:
			encoded = TensorHelper().encode_and_check(tokenizer, conversation)
			if encoded is not None:
				valid_prompts.append(encoded)

		max_length = max([len(tokenizer.encode(prompt)) for prompt in valid_prompts])

		model.resize_token_embeddings(len(tokenizer))

		dataset = RedditDataset(valid_prompts, tokenizer, max_length=max_length)

		train_size = int(0.9 * len(dataset))

		train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

		gc.collect()
		torch.cuda.empty_cache()

		args = dict(
			output_dir='./results',
			num_train_epochs=10,
			logging_steps=1,
			save_steps=500,
			per_device_train_batch_size=1,
			per_device_eval_batch_size=1,
			weight_decay=0.05,
			logging_dir='./logs',
			report_to='none',
			fp16=True,
			auto_find_batch_size=False,
			gradient_accumulation_steps=1,
			learning_rate=1e-4)

		training_args = TrainingArguments(**args)

		Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
				data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
											'attention_mask': torch.stack([f[1] for f in data]),
											'labels': torch.stack([f[0] for f in data])}).train()
