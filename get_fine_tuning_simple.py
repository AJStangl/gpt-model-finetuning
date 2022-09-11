import asyncio
import logging

import pandas

from src.tensor_encoding.tensor_encoding import TokenizerAdapter
from src.training_module.training_finetuning_collector import RedditFineTuningCollector


async def main():
	# subs = open("subreddits.txt", "r").read().split(",")
	subs = ['wallstreetbets', 'DoesAnybodyElse', 'whatisthisthing']
	logging.info(": Processing the following subreddits")
	for sub in subs:
		logging.info(f"=== Starting Collection For {sub} ===")
		await RedditFineTuningCollector()\
			.get_top_100(subreddit=sub, time_filter="month", output_file="coop_training.csv")
		logging.info(f"=== Subreddit completed {sub} ===")
	logging.info(f"=== Process Complete ===")

# def filter_token_length():
# 	helper: TokenizerAdapter = TokenizerAdapter(tokenizer=None)
# 	df = pandas.read_csv("coop_training.csv", encoding='utf-8')
# 	training_string_series = df["TrainingString"]
# 	training_string_series.where(lambda x: helper.token_length_appropriate(x), inplace=True)



if __name__ == '__main__':
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
	asyncio.run(main())