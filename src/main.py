import asyncio
import logging

from training_module.training_finetuning_collector import FineTuningDataCollector
import click


async def main(subreddit: str):
	if subreddit is None:
		logging.error("No subreddit present")
		return None
	collector: FineTuningDataCollector = FineTuningDataCollector()
	await collector.get_top_100(subreddit)

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	try:
		sub = "onlyfansadvice"
		asyncio.run(main(sub))
	except:
		logging.error("Specify sub reddit")
