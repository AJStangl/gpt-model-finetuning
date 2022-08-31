import logging
import datetime
from asyncpraw import Reddit


class RedditManager:
	def __init__(self):
		self.instance: Reddit = self.get_instance()

	# TODO MAKE THIS A CONFIGURATION
	def get_instance(self, bot_name: str = "ChadNoctorBot-GPT2") -> Reddit:
		logging.debug(f":: Initializing Reddit Praw Instance for {bot_name}")
		reddit = Reddit(site_name=bot_name)
		return reddit
