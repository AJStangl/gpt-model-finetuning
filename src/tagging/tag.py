import codecs
import logging
import re

import ftfy
from asyncpraw import Reddit
from asyncpraw.models import Submission, Comment
from asyncpraw.models.reddit.base import RedditBase


class Tagging:
	"""
	TODO: Describe this class
	"""

	_link_submission_start_tag = '<|sols|>'
	_selftext_submission_start_tag = '<|soss|>'

	_title_start_tag = '<|sot|>'
	_selftext_start_tag = '<|sost|>'

	_reply_start_tag = '<|sor|>'
	_reply_end_tag = '<|eor|>'

	_end_tag = '<|'

	def __init__(self, reddit: Reddit):
		self.reddit_instance = reddit

	# TODO: Replace with method that traverses through the a tree representation:
	# Submission -> Node[Comment[ReplyId]-ReplyId->Node[Comment[ReplyId]]
	# The current algo does it but uses PRAW to make these calls. It is a 1:1 with what we pass to our actual model but
	# construction of the training string is needlessly expensive. Mapping a method to get the training string is trivial
	# what is not is calculating the final token size to ensure we don't go over the model maximum
	async def collate_tagged_comment_history(self, loop_thing: RedditBase, to_level=12) -> str:
		"""
		Loop backwards (upwards in reddit terms) from the praw_thing through the comment up x times,
		tagging the content text in the same way as the training data is
		The resulting string will be passed to the model to generate a reply to
		*This section is customisable for your own bot and how it has been finetuned*
		Each <|tag|> behaves as metadata so the model knows the general writing style of
		titles, replies and so forth.
		"""
		counter = 0
		prefix = ''
		try:
			await loop_thing.load()
		except Exception as e:
			logging.error(f":: Error loading comment for collate_tagged_comment_history with error {e}")
			return prefix

		while loop_thing and counter < to_level:

			if isinstance(loop_thing, Submission):
				tagged_text = await self.tag_submission(loop_thing)
				prefix = tagged_text + prefix

				# can't go any higher than a submission, so break the loop
				break

			elif isinstance(loop_thing, Comment):
				# It's a comment
				tagged_text = await self.tag_comment(loop_thing)
				prefix = tagged_text + prefix
				loop_thing = await loop_thing.parent()
			counter += 1
		return prefix

	async def get_reply_tag(self, thing: RedditBase) -> str:
		"""
		Get the reply tag to use.
		The model will generate text after this reply tag.

		*This section is customisable for your own bot and how it has been fine-tuned
		"""
		try:
			if isinstance(thing, Comment):
				base: RedditBase = thing
				await base.load()

				submission_id = base.submission.id
				submission: Submission = await self.reddit_instance.submission(id=submission_id, fetch=True)
				await submission.load()

				# If the submission is the same as the responding bot use the <|soopr|> tag
				if isinstance(base, Comment) and submission.author == base.author:
					return '<|soopr|>'

				if isinstance(base, Comment):
					try:
						parent_of_parent = await self.get_parent_of_parent(base)
						await parent_of_parent.load()
					except Exception as e:
						logging.error(f":: Failed to get parent of parent. Returning default. Error {e}")
						return self._reply_start_tag

					if parent_of_parent.author == base.author:
						return '<|soocr|>'

			if isinstance(thing, Submission):
				return self._reply_start_tag

		except Exception as e:
			logging.info(f":: {e} in get_reply_tag")
			return self._reply_start_tag

		# It's just a straight reply
		return self._reply_start_tag

	def get_random_new_submission_tag(self, subreddit: str, use_reply_sense=True):
		import random
		# random is already seeded in reddit_io init
		random_value = random.random()

		tag = ''

		if random_value < 0:
			# Make a link (image) post
			tag += '<|sols'
		else:
			# Make a text post
			tag += '<|soss'

		if use_reply_sense:
			tag += f' r/{subreddit}|>'
		else:
			tag += '|>'

		return tag + self._title_start_tag

	@staticmethod
	async def tag_submission(submission: Submission):
		tagged_text = ""

		try:
			await submission.load()
		except Exception as e:
			logging.error(f":: Failed to load submission in tag_submission with error\n{e}")
			return tagged_text

		if not isinstance(submission, Submission):
			return tagged_text

		if submission.is_self:
			tagged_text += "<|soss"
		else:
			tagged_text += "<|sols"

		tagged_text += f" r/{submission.subreddit}|>"

		# prepend the tagged text
		if submission.is_self:

			selftext = submission.selftext

			if hasattr(submission, 'poll_data'):
				for option in submission.poll_data.options:
					selftext += f" - {option.text}"

			# selftext submission
			tagged_text += f"<|sot|>{submission.title}<|eot|><|sost|>{selftext}<|eost|>"

		else:
			# it's a link submission
			tagged_text += f"<|sot|>{submission.title}<|eot|><|sol|><|eol|>"

		return tagged_text

	async def tag_comment_with_sub(self, comment: Comment, submission: Submission) -> str:
		try:
			if submission.author == comment.author:
				return f'<|soopr u/{comment.author}|>{comment.body}<|eoopr|>'

			parent_parent = await self.get_parent_of_parent(comment)

			await parent_parent.load()

			if parent_parent.author == comment.author:
				return f'<|soocr u/{comment.author}|>{comment.body}<|eoocr|>'
			else:
				return f'<|sor u/{comment.author}|>{comment.body}<|eor|>'

		except Exception as e:
			logging.error(f"{e} in tag_comment")
			return f'<|sor|>{comment.body}<|eor|>'

	async def tag_comment(self, comment: Comment) -> str:
		try:
			await comment.load()
			submission_id = comment.submission.id

			submission: Submission = await self.reddit_instance.submission(id=submission_id)
			await submission.load()

			if submission.author == comment.author:
				return f'<|soopr u/{comment.author}|>{comment.body}<|eoopr|>'

			parent_parent = await self.get_parent_of_parent(comment)

			await parent_parent.load()

			if parent_parent.author == comment.author:
				return f'<|soocr u/{comment.author}|>{comment.body}<|eoocr|>'
			else:
				return f'<|sor u/{comment.author}|>{comment.body}<|eor|>'

		except Exception as e:
			logging.error(f"{e} in tag_comment")
			return f'<|sor|>{comment.body}<|eor|>'

	@staticmethod
	def tag_message(thing, use_reply_sense=True):

		tagged_text = ""

		if not thing.parent_id:
			# If parent_id property is None then it is the first message of the chain
			tagged_text += f'<|sot>{thing.subject}<|eot|>'

		if use_reply_sense:
			tagged_text += f'<|soocr|>{thing.body}<|eoocr|>'
		else:
			tagged_text += f'<|sor|>{thing.body}<|eor|>'

		return tagged_text

	def extract_reply_from_generated_text(self, prompt: str, generated_text: str) -> dict:

		if prompt is None or generated_text is None:
			return {}

		# remove any cruft
		generated_text = generated_text.replace('&amp;#x200B;\n', '')

		# find the first instance of the end-of-comment tag, starting from the end of the prompt
		index_of_truncate = generated_text.find(self._end_tag, len(prompt))

		if index_of_truncate == -1:
			# the original truncate tag couldn't be found,
			# but we'll still try and truncate the string at the last line break (end of paragraph)
			# so that the text still looks clean.
			index_of_truncate = generated_text.rfind("\\n")

		if index_of_truncate == -1:
			# in case trained model do not output tags and put lot !!!!! at the end,
			# This change allows this messages without need of end tags
			index_of_truncate = generated_text.find("!!!!")

		if index_of_truncate == -1:
			# still nothing could be found so just skip this one
			# if this is hit often, increase the length of the generated text
			return {}

		# extract the text from between the prompt and the truncate point
		reply_body = generated_text[len(prompt):index_of_truncate]
		if reply_body:
			return {'body': self._decode_generated_text(reply_body)}

		# Return nothing
		return {}

	def extract_title_from_generated_text(self, generated_text):

		idx_title_start = generated_text.find(self._title_start_tag)

		idx_title_end = generated_text.find(self._end_tag, (idx_title_start + len(self._title_start_tag)))

		if idx_title_start == -1 or idx_title_end == -1:
			# There must be at least a complete title to make a submission
			return None

		title_text = generated_text[idx_title_start + len(self._title_start_tag):idx_title_end]

		if 0 < len(title_text) < 300:
			# Validate the title length is within reddit's range
			return self._decode_generated_text(title_text)

	def extract_selftext_from_generated_text(self, generated_text):

		idx_st_start = generated_text.find(self._selftext_start_tag)

		idx_st_end = generated_text.find(self._end_tag, (idx_st_start + len(self._selftext_start_tag)))

		if idx_st_start == -1 or idx_st_end == -1:
			return None

		selftext_text = generated_text[idx_st_start + len(self._selftext_start_tag):idx_st_end]

		return self._decode_generated_text(selftext_text)

	def extract_submission_from_generated_text(self, generated_text):

		return_dict = {}

		if generated_text is None:
			return {}

		# remove any cruft
		generated_text = generated_text.replace('&amp;#x200B;\n', '')

		title = self.extract_title_from_generated_text(generated_text)

		if not title:
			return {}
		else:
			# The title is ok, add it to the dict to return
			return_dict['title'] = title

		selftext = self.extract_selftext_from_generated_text(generated_text)

		if selftext:
			return_dict['selftext'] = selftext

		return return_dict

	@staticmethod
	def remove_tags_from_string(input_string):
		# Removes any <|sor u/user|>, <|sost|> etc from a string
		return re.sub(r'(\<\|[\w\/ ]*\|\>)', ' ', input_string).strip()

	@staticmethod
	def _decode_generated_text(text):
		return ftfy.fix_text(codecs.decode(text, "unicode_escape"))

	@staticmethod
	def remove_username_mentions_from_string(string: str, username: str) -> str:
		regex = re.compile(fr"u\/{username}(?!\|\>)", re.IGNORECASE)
		return regex.sub('', string)

	async def get_parent_of_parent(self, reddit_base: RedditBase) -> RedditBase:
		# Guard for a base object coming is a submission.
		if self.is_submission(reddit_base):
			return reddit_base

		# cast to comment - compiler hint
		base: RedditBase = reddit_base
		try:
			await base.load()
		except Exception as e:
			logging.error(f"Error attempting to load base object {e} in get_parent_of_parent")
			return base

		# get the parent for the first item
		try:
			parent = await base.parent()
			await parent.load()
		except Exception as e:
			logging.error(f"Error attempting to get the parent from base {e} in get_parent_of_parent")
			return base

		# if it's a submission then return the submission
		if self.is_submission(parent):
			return parent

		# Then move up one more time to get the parent's parent
		try:
			parent_parent = await parent.parent()
			await parent_parent.load()
		except Exception as e:
			logging.error(f"Error Attempting to get parent.parent from base {e} in get_parent_of_parent")
			return parent

		if self.is_submission(parent_parent):
			await parent_parent.load()
			return parent_parent
		else:
			return parent

	@staticmethod
	def is_submission(reddit_base: RedditBase) -> bool:
		return isinstance(reddit_base, Submission)
