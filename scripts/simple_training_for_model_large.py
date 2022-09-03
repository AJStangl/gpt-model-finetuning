import gc
import logging
import os

import pandas
import torch
from src.tensor_encoding.tensor_encoding import TensorHelper
from src.datasets.reddit_dataset import RedditDataset
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer
import json
import gc

import pandas
import torch
from torch.utils.data import random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

from src.datasets.reddit_dataset import RedditDataset
from src.tensor_encoding.tensor_encoding import TensorHelper


def main():
	logging.basicConfig(level=logging.INFO)

	out_dir = "mega_bot_2"
	df = pandas.read_json("D:\\code\\repos\\quick_training\\final_training.json",orient='split')

	prompts = list(df['Prompt'])

	tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
											  bos_token='<|startoftext|>',
											  eos_token='<|endoftext|>',
											  pad_token='<|pad|>')

	logging.info(f"Saving tokenizer to {out_dir}")
	tokenizer.save_pretrained(f'{out_dir}')

	logging.info(f"Loading model from {out_dir}")
	model = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda()

	valid_prompts = []
	logging.info(":: Checking encodings...")
	for prompt in prompts:
		# encoded = TensorHelper.encode_and_check(tokenizer, prompt)
		if prompt is not None:
			valid_prompts.append(prompt)

	logging.info(f":: Checking for max length")
	max_length = max([len(tokenizer.encode(prompt)) for prompt in valid_prompts])
	logging.info(f":: Max lenth is {max_length}")

	logging.info(f":: Resising embeddings")
	model.resize_token_embeddings(len(tokenizer))

	logging.info(f":: Loading Dataset")
	dataset = RedditDataset(valid_prompts, tokenizer, max_length=max_length)

	train_size = int(0.9 * len(dataset))

	logging.info(f":: Splitting to train and eval sets")
	train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

	logging.info(f":: Train Set: {len(train_dataset)}\t|\tEval Set: {len(eval_dataset)}")
	gc.collect()

	torch.cuda.empty_cache()

	args = dict(
		output_dir= out_dir,
		num_train_epochs=10,
		logging_steps=100,
		save_steps=1000,
		# Allow the computer to be smarter than you...
		# per_device_train_batch_size=2,
		# per_device_eval_batch_size=2,
		weight_decay=0.05,
		logging_dir='./logs',
		report_to='none',
		fp16=True,
		auto_find_batch_size=True,
		gradient_accumulation_steps=10,
		learning_rate=1e-4)

	training_args = TrainingArguments(**args)

	Trainer(model=model, args=training_args, train_dataset=train_dataset,
			eval_dataset=eval_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
																   'attention_mask': torch.stack([f[1] for f in data]),
																   'labels': torch.stack([f[0] for f in data])
																   }).train()
if __name__ == '__main__':
	main()
