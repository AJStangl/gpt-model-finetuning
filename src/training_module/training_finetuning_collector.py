from logging import getLogger
from typing import Tuple, AsyncGenerator

import pandas
from asyncpraw import Reddit
from asyncpraw.models import Subreddit, Comment, Submission
from asyncpraw.models.comment_forest import CommentForest
from asyncpraw.models.reddit.base import RedditBase

from src.reddit.reddit_manager import RedditManager
from src.tagging.tag import Tagging
from src.training_module.data_model.training_row import TrainingRow

logger = getLogger("FineTuningDataCollector")


class FineTuningDataCollector:
	def __init__(self):
		self.__reddit_manager: RedditManager = RedditManager()
		self.__instance: Reddit = self.__reddit_manager.get_instance()
		self.__limit: int = 100

	async def get_top_100(self, subreddit: str, time_filter: str = "month"):
		output_file: str = "training.csv"
		df = self._load_previous_dataframe(output_file)
		tagging: Tagging = Tagging(self.__instance)
		subreddit: Subreddit = await self.__instance.subreddit(subreddit)
		i = 0
		lines_written = 0
		async for submission in subreddit.top(time_filter=time_filter, limit=self.__limit):
			# submission.link_flair_text.strip().lower()
			logger.info(f"{i}/{self.__limit} Submissions completed")
			async for comments in self._get_all_comments(submission):
				for comment in comments:
					if self._comment_exists_in_dataframe(df, comment.id):
						continue

					result = await tagging.collate_tagged_comment_history(comment)

					parent_id, parent_author, parent_body = await self._set_parent_information(comment)

					temp_df = TrainingRow() \
						.set_subreddit_information(subreddit) \
						.set_comment_information(comment) \
						.set_submission_information(submission) \
						.set_parent_information(parent_author=parent_author, parent_id=parent_id, parent_body=parent_body) \
						.set_training_text(result) \
						.to_df()

					df = pandas.concat([df, temp_df], ignore_index=True)
					df.to_csv(output_file)
					lines_written += 1
					logger.info(f"{lines_written} new lines written")
			i += 1
		await self.__instance.close()

	@staticmethod
	async def _set_parent_information(comment: Comment) -> Tuple[str, str, str]:
		parent: RedditBase = await comment.parent()

		if isinstance(parent, Submission):
			body = f"{parent.title} {parent.selftext}"
			return parent.id, parent.author, body

		if isinstance(parent, Comment):
			await parent.load()
			return parent.id, parent.author, parent.body

		return "", "", ""

	@staticmethod
	async def _get_all_comments(submission: Submission) -> AsyncGenerator:
		comments: CommentForest = await submission.comments()
		await comments.replace_more(limit=None)
		yield comments.list()

	@staticmethod
	def _load_previous_dataframe(output_file: str) -> pandas.DataFrame:
		try:
			df = pandas.read_csv(output_file)
			return df
		except FileNotFoundError:
			return pandas.DataFrame(columns=TrainingRow.get_header_columns())

	@staticmethod
	def _comment_exists_in_dataframe(dataframe: pandas.DataFrame, commentId) -> bool:
		return commentId in set(dataframe['CommentId'])
