from logging import getLogger
from typing import Optional

from transformers import GPT2Tokenizer, PreTrainedTokenizer

logger = getLogger("TensorHelper")


class TensorHelper:
	"""
	Provides an interface to tokenization related activities. Intended to be extended on from this type.
	"""

	"""
	Private constant for maximum acceptable length of GPT-2 models. Perhaps this requires further abstraction. 
	"""
	MAX_TOKEN_LIMIT: int = 1024

	def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None):
		self.__tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")

	def token_length_appropriate(self, prompt) -> bool:
		"""
		Ensures that the total number of encoded tokens is within acceptable limits.
		:param tokenizer: An instance of the tokenizer being used.
		:param prompt: UTF-8 Text that is assumed to have been processed.
		:return: True if acceptable.
		"""
		tokens = self.__tokenizer.tokenize(prompt)
		if len(tokens) > self.MAX_TOKEN_LIMIT:
			logger.debug(f":: Tokens for model input is > {1024}. Skipping input")
			return False
		else:
			return True
