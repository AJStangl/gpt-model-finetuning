import csv
from logging import getLogger
from typing import Tuple
import torch

import pandas
from simpletransformers.language_modeling import LanguageModelingModel

logger = getLogger("ModelFineTuner")


class ModelFineTuner:
	bot_label: str
	training_arguments: dict

	def __init__(self, bot_label):
		self.bot_label: str = bot_label
		self.training_arguments: dict = self.get_training_arguments(self.bot_label)

	def get_model(self) -> LanguageModelingModel:
		model = LanguageModelingModel(model_type=self.training_arguments['model_type'],
									  model_name=self.training_arguments['model_name'],
									  args=self.training_arguments,
									  use_cuda=True, cuda_device="1")
		return model

	def get_existing_model(self, model_path: str) -> LanguageModelingModel:
		model = LanguageModelingModel(self.training_arguments['model_type'],
									  model_path,
									  args=self.training_arguments,
									  use_cuda=True,
									  cuda_device="1")

		return model

	def train_model(self, model, train_file: str, eval_file):
		model.train_model(train_file=train_file, eval_file=eval_file, args=self.training_arguments)

	@staticmethod
	def generate_text_training_data(training_data: str, bot_label: str) -> Tuple[str, str]:
		train_text_file = f'{bot_label}_train.txt'
		eval_text_file = f'{bot_label}_eval.txt'

		subreddits = ["onlyfansadvice", "CreatorsAdvice"]
		df = pandas.read_csv(training_data)

		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

		# Drop unnamed columns
		df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

		# Clear duplicates
		df = df.T.drop_duplicates().T

		filtered = df.where(df['CommentBody'] != '[deleted]')
		filtered.dropna(inplace=True, subset=['Subreddit'])

		filtered = filtered.where(filtered['CommentBody'] != '[removed]')
		filtered.dropna(inplace=True, subset=['Subreddit'])

		filtered = filtered.where(filtered['Subreddit'].isin(subreddits))
		filtered.dropna(inplace=True, subset=['Subreddit'])

		train_df = filtered.sample(frac=.9)
		train_ids = list(train_df['CommentId'])

		eval_df = filtered.where(~filtered['CommentId'].isin(train_ids))
		eval_df.dropna(inplace=True, subset=['Subreddit'])

		training_text = train_df["TrainingString"].apply(lambda x: x.replace('\n', '\\n'))
		eval_text = eval_df["TrainingString"].apply(lambda x: x.replace('\n', '\\n'))

		training_text.to_csv(train_text_file, index=False, header=False, line_terminator='\n', encoding="utf-8", escapechar='\\', quoting=csv.QUOTE_NONE)
		eval_text.to_csv(eval_text_file, index=False, header=False, line_terminator='\n', encoding="utf-8", escapechar='\\', quoting=csv.QUOTE_NONE)

		return train_text_file, eval_text_file

	@staticmethod
	def get_training_arguments(bot_label: str) -> dict:
		model_args = {
			"model_type": "gpt2",
			"model_name": "gpt2-medium",
			"overwrite_output_dir": True,
			"learning_rate": 1e-4,
			# larger batch sizes will use more training data but consume more ram
			# accumulation steps
			"gradient_accumulation_steps": 50,

			# Use text because of grouping by reddit submission
			"dataset_type": "simple",
			# Sliding window will help it manage very long bits of text in memory
			"sliding_window": True,
			"max_seq_length": 1024,
			"max_steps": -1,

			"mlm": False,  # has to be false for gpt-2

			"evaluate_during_training": True,
			# default 2000, will save by default at this step.
			# "evaluate_during_training_steps": 2000,
			"use_cached_eval_features": True,
			"evaluate_during_training_verbose": True,

			# don't save optimizer and scheduler we don't need it
			"save_optimizer_and_scheduler": False,
			# Save disk space by only saving on checkpoints
			"save_eval_checkpoints": True,
			"save_model_every_epoch": False,
			# disable saving each step to save disk space
			"save_steps": -1,

			"output_dir": f"{bot_label}/",
			"best_model_dir": f"{bot_label}/best_model",
		}
		if 'K80' in torch.cuda.get_device_name(0):
			# Most of the time we'll only get a K80 on free Colab
			model_args['train_batch_size'] = 1
			# Need to train for multiple epochs because of the small batch size
			model_args['num_train_epochs'] = 6
			model_args["gradient_accumulation_steps"] = 100
			# Save every 3000 to conserve disk space
			model_args["evaluate_during_training_steps"] = int(3000 / model_args["gradient_accumulation_steps"])

		elif 'T4' in torch.cuda.get_device_name(0):
			# You may get a T4 if you're using Colab Pro
			# larger batch sizes will use more training data but consume more ram
			model_args['train_batch_size'] = 8
			# On Tesla t4 we can train for steps rather than epochs because of the batch size
			model_args["max_steps"] = -1
			# default 3000, will save by default at this step.
			model_args["evaluate_during_training_steps"] = 3000

		return model_args
