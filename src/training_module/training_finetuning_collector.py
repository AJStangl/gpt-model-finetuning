from logging import getLogger
from typing import Tuple, AsyncGenerator, Optional

import pandas
from asyncpraw import Reddit
from asyncpraw.models import Subreddit, Comment, Submission
from asyncpraw.models.comment_forest import CommentForest
from asyncpraw.models.reddit.base import RedditBase

from src.reddit.reddit_manager import RedditManager
from src.tagging.tag import Tagging
from src.training_module.data_model.training_row import TrainingRow

logger = getLogger("FineTuningDataCollector")


class FineTuningCollector:
	"""
	TODO
	"""
	def __init__(self):
		"""
		TODO
		"""
		pass



class RedditFineTuningCollector(FineTuningCollector):
	"""
	TODO:
	"""
	def __init__(self):
		"""
		TODO:
		"""
		super().__init__()
		self.__reddit_manager: RedditManager = RedditManager()
		self.__instance: Reddit = self.__reddit_manager.get_instance()
		self.__limit: int = 100
		self.output_file = "training.csv"

	# TODO: Implement get_*_100 methods and implement correct search criteria
	async def get_top_100(self, subreddit: str, time_filter: str = "month"):
		# TODO: Make configurable (maybe)
		df = self._load_previous_dataframe(self.output_file)
		tagging: Tagging = Tagging(self.__instance)
		subreddit: Subreddit = await self.__instance.subreddit(subreddit)
		i = 0
		lines_written = 0
		# TODO: Make async iterator
		async for submission in subreddit.top(time_filter=time_filter, limit=self.__limit):
			logger.info(f"{i}/{self.__limit} Submissions completed")
			async for comments in self._get_all_comments(submission):
				for comment in comments:
					# TODO: Introduce configuration filter for this
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

					df.to_csv(self.output_file)
					lines_written += 1
					logger.info(f"{lines_written} new lines written")
			i += 1
		await self.__instance.close()

	def _load_previous_dataframe(self, output_file: str) -> pandas.DataFrame:
		"""
		TODO:
		:param output_file:
		:return:
		"""
		try:
			df = pandas.read_csv(output_file)
			df = self._clean_previous_dataframe(data_frame=df)
			return df
		except FileNotFoundError:
			return pandas.DataFrame(columns=TrainingRow.get_header_columns())

	@staticmethod
	async def _set_parent_information(comment: Comment) -> Tuple[str, str, str]:
		"""

		:param comment:
		:return:
		"""
		parent: RedditBase = await comment.parent()

		if isinstance(parent, Submission):
			body = f"{parent.title} {parent.selftext}"
			return parent.id, parent.author, body

		if isinstance(parent, Comment):
			await parent.load()
			return parent.id, parent.author, parent.body

		return "", "", ""

	@staticmethod
	async def _get_all_comments(submission: Submission, limit: Optional[int] = None) -> AsyncGenerator:
		"""

		:param submission:
		:param limit:
		:return:
		"""
		comments: CommentForest = await submission.comments()
		await comments.replace_more(limit=limit)
		yield comments.list()

	@staticmethod
	def _comment_exists_in_dataframe(dataframe: pandas.DataFrame, commentId) -> bool:
		"""
		TODO:
		:param dataframe:
		:param commentId:
		:return:
		"""
		return commentId in set(dataframe['CommentId'])

	@staticmethod
	def _clean_previous_dataframe(data_frame: pandas.DataFrame) -> pandas.DataFrame:
		"""
		TODO
		:param data_frame:
		:return:
		"""
		# data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^Unnamed')]
		# # Drop unnamed columns
		# data_frame.drop(data_frame.columns[data_frame.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
		#
		# # Clear duplicates
		# data_frame = data_frame.T.drop_duplicates().T
		return data_frame
