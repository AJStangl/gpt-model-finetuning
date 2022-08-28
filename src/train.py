import pandas
import torch
from simpletransformers.language_modeling import LanguageModelingModel
import os


def main():
	# df = pandas.read_csv("training.csv")
	bot_label = 'only_fans_bot_2'
	# train_text_file = "train.txt"
	# eval_text_file = "eval.txt"
	#
	# train_df = df.sample(frac=.9)
	# train_ids = list(train_df["CommentId"])
	# eval_df = df.where(~df["CommentId"].isin(train_ids)).dropna()
	#
	# training_text = train_df["TrainingString"].apply(lambda x: x.replace('\n', '\\n'))
	# eval_text = eval_df["TrainingString"].apply(lambda x: x.replace('\n', '\\n'))
	#
	# training_text.to_csv(train_text_file, index=False, header=False, line_terminator='\n', encoding="utf-8")
	# eval_text.to_csv(eval_text_file, index=False, header=False, line_terminator='\n', encoding="utf-8")
	#
	# with open(eval_text_file, 'a') as f:
	# 	text_file = f.read()
	# 	text_file.replace('"', '')
	# 	f.close()
	#
	# with open(train_text_file, 'a') as f:
	# 	text_file = f.read()
	# 	text_file.replace('"', '')
	# 	f.close()

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
		model_args["evaluate_during_training_steps"] = 3000,

	# Check to see if a model already exists for this bot_label
	resume_training_path = f"/{bot_label}/best_model/"

	if os.path.exists(resume_training_path):
		# A model path already exists. So we'll attempt to resume training starting fom the previous best_model.
		model_args['output_dir'] = resume_training_path
		model_args['best_model_dir'] = f"{resume_training_path}/resume_best_model/"
		model = LanguageModelingModel(model_args['model_type'], resume_training_path, args=model_args, use_cuda=True, cuda_device="1")
	else:
		model = LanguageModelingModel(model_type=model_args['model_type'], model_name=model_args['model_name'], args=model_args, use_cuda=True, cuda_device="1")

	model.train_model(train_file="D:\\code\\repos\\quick_training\\only_fans_2_train.txt", eval_file="D:\\code\\repos\\quick_training\\only_fans_2_eval.txt", args=model_args)


if __name__ == '__main__':
	main()
