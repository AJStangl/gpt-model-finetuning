from src.tensor_encoding.tensor_encoding import TokenizerAdapter


class RedditDataFilter:
	def __init__(self):
		self.tensor_helper = TokenizerAdapter()


	def has_valid_line(self, input: str, tokenizer) -> bool:
		black_list = ["**NO SIGN**", "**Image Stats:**", "**INCOMPLETE MEAT TUBE**", "[removed]", "[deleted]",
					  'Unfortunately, your post was removed for the following reason(s)']

		for line in black_list:
			if input.__contains__(line):
				return False

			return self.tensor_helper.token_length_appropriate(line)

	def filter_text_filter(self, file_path):
		# TODO: Pandas shit
		pass
