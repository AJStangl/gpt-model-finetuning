import html

import pandas
from asyncpraw.models import Submission, Comment, Subreddit


class TrainingRow(object):
	Subreddit: str
	SubmissionId: str
	ParentId: str
	ParentAuthor: str
	ParentBody: str
	CommentId: str
	CommentBody: str
	CommentAuthor: str
	TrainingString: str

	@staticmethod
	def get_header_columns() -> [str]:
		return ["Subreddit", "SubmissionId", "ParentId", "ParentAuthor", "ParentBody", "CommentId", "CommentBody", "TrainingString"]

	def set_subreddit_information(self, subreddit: Subreddit):
		self.Subreddit = subreddit.display_name
		return self

	def set_submission_information(self, submission: Submission):
		self.SubmissionId = submission.id
		return self

	def set_parent_information(self, parent_id: str, parent_author: str, parent_body: str):
		self.ParentAuthor = parent_author
		self.ParentId = parent_id
		self.ParentBody = parent_body
		return self

	def set_comment_information(self, comment: Comment):
		self.CommentId = comment.id
		self.CommentBody = comment.body
		self.CommentAuthor = comment.author
		return self

	def set_training_text(self, training: str):
		text = self.clean_text(training)
		self.TrainingString = text
		return self

	def to_df(self) -> pandas.DataFrame:
		return pandas.DataFrame(self.__dict__, index=[0])

	@staticmethod
	def clean_text(text) -> str:
		if text is None:
			text = ""
		# have to unescape it twice, for reason I don't fully understand
		text = html.unescape(text)
		text = html.unescape(text)
		# Strip and whitespace off of the end
		text = text.strip()
		return text
