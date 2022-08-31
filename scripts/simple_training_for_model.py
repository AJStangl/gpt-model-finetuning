import gc
import os

import pandas
import torch
from src.tensor_encoding.tensor_encoding import TensorHelper
from src.datasets.reddit_dataset import RedditDataset
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Don't at me... I got many things my gpu's do

if __name__ == '__main__':

	tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

	tokenizer.save_pretrained('./results')

	model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

	df = pandas.read_csv("training.csv")

	conversations = df['TrainingString']

	valid_prompts = []
	for conversation in conversations:
		encoded = TensorHelper.encode_and_check(tokenizer, conversation)
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

	Trainer(model=model, args=training_args, train_dataset=train_dataset,
			eval_dataset=eval_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
																   'attention_mask': torch.stack([f[1] for f in data]),
																   'labels': torch.stack([f[0] for f in data])
																   }).train()
