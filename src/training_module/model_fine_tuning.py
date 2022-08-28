from logging import getLogger
from typing import Tuple

import pandas
from simpletransformers.language_modeling import LanguageModelingModel

logger = getLogger("ModelFineTuner")


class ModelFineTuner:
	bot_label: str
	training_arguments: dict

	def __init__(self, bot_label):
		self.bot_label: str = bot_label
		self.training_arguments: dict = {
			"model_type": "gpt2",
			"model_name": "gpt2-medium",
			"overwrite_output_dir": True,
			"learning_rate": 1e-4,
			# larger batch sizes will use more training data but consume more ram
			# accumulation steps
			"gradient_accumulation_steps": 1,

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

			"output_dir": f"{self.bot_label}/",
			"best_model_dir": f"{self.bot_label}/best_model",
		}

	def get_model(self) -> LanguageModelingModel:
		model = LanguageModelingModel(model_type=self.training_arguments['model_type'],
									  model_name=self.training_arguments['model_name'], args=self.training_arguments,
									  use_cuda=True, cuda_device="1")
		return model

	def get_existing_model(self, model_path: str) -> LanguageModelingModel:
		model = LanguageModelingModel(self.training_arguments['model_type'], model_path, args=self.training_arguments,
									  use_cuda=True,
									  cuda_device="1")

		return model

	def train_model(self, model, train_file: str, eval_file):
		model.train_model(train_file=train_file, eval_file=eval_file, args=self.training_arguments)

	@staticmethod
	def generate_text_training_data(training_data: str, bot_label: str) -> Tuple[str, str]:
		df = pandas.read_csv(training_data)
		train_text_file = f"{bot_label}_train.txt"
		eval_text_file = f"{bot_label}_eval.txt"

		train_df = df.sample(frac=.9)
		train_ids = list(train_df["CommentId"])
		eval_df = df.where(~df["CommentId"].isin(train_ids)).dropna()

		training_text = train_df["TrainingString"].apply(lambda x: x.replace('\n', '\\n'))
		eval_text = eval_df["TrainingString"].apply(lambda x: x.replace('\n', '\\n'))

		training_text.to_csv(train_text_file, index=False, header=False, line_terminator='\n', encoding="utf-8")
		eval_text.to_csv(eval_text_file, index=False, header=False, line_terminator='\n', encoding="utf-8")

		with open(eval_text_file, 'a') as f:
			text_file = f.read()
			text_file.replace('"', '')
			f.close()

		with open(train_text_file, 'a') as f:
			text_file = f.read()
			text_file.replace('"', '')
			f.close()

		return text_file, eval_text
