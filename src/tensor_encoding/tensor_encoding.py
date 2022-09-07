from logging import getLogger

logger = getLogger("TensorHelper")


class TensorHelper:
	"""
	TODO: Class description
	"""
	MAX_TOKEN_LIMIT: int = 1024

	def encode_and_check(self, tokenizer, prompt) -> bool:
		"""
		TODO: Document this stupid fucking function that is the key to not having your model blow the fuck up.
		:param tokenizer:
		:param prompt:
		:return:
		"""
		tokens = tokenizer.tokenize(prompt)
		if len(tokens) > self.MAX_TOKEN_LIMIT:
			logger.debug(f":: Tokens for model input is > {1024}. Skipping input")
			return False
		else:
			return True
