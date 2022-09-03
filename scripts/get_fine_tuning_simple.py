import asyncio
import logging
from src.training_module.training_finetuning_collector import RedditFineTuningCollector


async def main():
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	subs = open("subreddits.txt", "r").read().split(",")

	logging.info(": Processing the following subreddits")
	for sub in subs:
		logging.info(f"=== Starting Collection For {sub} ===")
		await RedditFineTuningCollector().get_top_100(subreddit=sub, time_filter="month", output_file="training.json")
		logging.info(f"=== Subreddit completed {sub} ===")
	logging.info(f"=== Process Complete ===")


if __name__ == '__main__':
	asyncio.run(main())