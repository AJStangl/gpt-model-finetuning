from typing import Optional


class TensorHelper:

	@staticmethod
	def encode_and_check(tokenizer, prompt) -> Optional[str]:
		prompt_text = prompt
		tokens = tokenizer.tokenize(prompt_text)
		if len(tokens) > 1024:
			print(f":: Tokens for model input is > {1024}. Skipping input")
			return None
		encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
		h_tensor = encoded_prompt.H
		t_tensor = encoded_prompt.T
		h = [item for item in h_tensor]
		t = [item for item in t_tensor]
		if len(h) == len(t):
			return prompt_text
		else:
			print(f":: Size of T Tensor {len(h)} is not equal to H tensor {len(h)}. Skipping model Generation")
			return None
