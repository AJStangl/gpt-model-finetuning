import logging
from logging import getLogger

import asyncclick as click

from src.text_generation.model_text_generator import ModelTextGenerator
from src.training_module.model_fine_tuning import ModelFineTuner
from src.training_module.training_finetuning_collector import FineTuningDataCollector


@click.group()
def cli1():
	pass


@click.group()
def cli2():
	pass
@click.group()
def cli3():
	pass

@cli1.command()
@click.option("--sub", prompt='specify a single subreddit name')
async def get_fine_tuning(sub: str):
	"""Command for downloading finetuning data"""
	collector: FineTuningDataCollector = FineTuningDataCollector()
	await collector.get_top_100(subreddit=sub)

@cli2.command()
@click.option("--bot-label", prompt='specify a bot label for bot', default='generic')
@click.option("--data", prompt='datafile for creating eval and train data', default='training.csv')
async def train_model(bot_label: str, data: str):
	"""Command for running fine-tuning for gpt-2 model"""
	model_fine_tuner = ModelFineTuner(bot_label)

	# train_file, eval_file = model_fine_tuner.generate_text_training_data(data, bot_label)
	train_file = "D:\\code\\repos\\quick_training\\only_fans_2_train.txt"
	eval_file = "D:\\code\\repos\\quick_training\\only_fans_2_eval.txt"

	model = model_fine_tuner.get_model()

	model_fine_tuner.train_model(model, train_file=train_file, eval_file=eval_file)

@cli3.command()
async def gen_text():
	model_path: str = "D:\\code\\repos\\quick_training\\only_fans_bot_2\\best_model"
	text_generator: ModelTextGenerator = ModelTextGenerator()
	text_generator.generate_text_with_no_wrapper(model_path, "<|sols r/ExtremeCombat|><|sot|>How are you doing today?<|eot|><|sol|>It feels nice out right?<|eol|><|sor|>")

cli = click.CommandCollection(sources=[cli1, cli2, cli3])

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	cli(_anyio_backend="asyncio")
