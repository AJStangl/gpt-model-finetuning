import logging

import asyncclick as click

from src.text_generation.model_text_generator import ModelTextGenerator
from src.training_module.model_fine_tuning import ModelFineTuner
from src.training_module.training_finetuning_collector import RedditFineTuningCollector


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
@click.option("--sub", prompt='specify a single subreddit name', default='')
async def download_data(sub: str):
	"""Command for downloading fine-tuning data"""
	collector: RedditFineTuningCollector = RedditFineTuningCollector()
	subs = sub.split(",")
	for sub in subs:
		await collector.get_top_100(subreddit=sub)


@cli2.command()
@click.option("--bot-label", prompt='specify a bot label for bot', default='generic')
@click.option("--train_file", prompt='training file path', default='train.txt')
@click.option("--eval_file", prompt='eval file path', default='eval.txt')
async def train_model(bot_label: str, train_file: str, eval_file: str):
	"""Command for running fine-tuning for gpt-2 model"""
	model_fine_tuner = ModelFineTuner(bot_label)

	model = model_fine_tuner.get_gpt2_model()

	model_fine_tuner.train_model(model, train_file=train_file, eval_file=eval_file)


@cli3.command()
@click.option("--model_path", prompt='specify a model path', default='D:\\models\\large_pablo_bot')
@click.option("--prompt", prompt='specify a prompt for model', default='>>>')
async def gen_text(model_path: str, prompt: str):
	"""command for generating text from a model"""
	text_generator: ModelTextGenerator = ModelTextGenerator()
	text_generator.generate_text_with_no_wrapper(model_path, prompt)

cli = click.CommandCollection(sources=[cli1, cli2, cli3])

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	cli(_anyio_backend="asyncio")
