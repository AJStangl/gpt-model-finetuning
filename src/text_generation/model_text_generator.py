import torch
import time
from logging import getLogger

from numpy import unicode
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logger = getLogger("ModelTextGenerator")

class ModelTextGenerator:
	def __init__(self):
		self.text_generation_parameters = {
			'max_length': 1024,
			'num_return_sequences': 1,
			'prompt': None,
			'temperature': 0.8,
			'top_k': 40,
			'top_p': .8,
			'do_sample': True,
			'repetition_penalty': 1.08,
			'stop_token': '<|endoftext|>'
		}

	def generate_text_with_no_wrapper(self, model_path: str, prompt_text: str, cuda_device: int = 1) -> str:
		start_time = time.time()
		try:
			device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

			tokenizer = GPT2Tokenizer.from_pretrained(model_path)

			encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

			if len(encoded_prompt.data[0]) > self.text_generation_parameters['max_length']:
				logger.info(
					f":: Size of Tensor {encoded_prompt.data[0]} > {self.text_generation_parameters['max_length']}. Rejecting Attempt to Process Input")
				return None

			generation_prompt = tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")

			generation_prompt = generation_prompt.to(device)

			model = GPT2LMHeadModel.from_pretrained(model_path)

			model = model.to(device)
			output_sequences = model.generate(
				inputs=generation_prompt['input_ids'],
				max_length=1024,
				min_length=100,
				do_sample=True,
				top_k=40,
				temperature=0.8,
				repetition_penalty=1.08,
				attention_mask=generation_prompt['attention_mask'],
				stop_token='<|endoftext|>',
			)
			text_generations = []

			for i in range(self.text_generation_parameters['num_return_sequences']):
				decoded_text = tokenizer.decode(output_sequences[i], skip_special_tokens=False)
				if decoded_text in ['[removed]'] or decoded_text == "":
					raise Exception("Text No Good Try Again!")
				text_generations.append(decoded_text)
				decoded_text.replace(prompt_text, "")
				print(f"Generated {i}: {decoded_text}")

			end_time = time.time()
			duration = round(end_time - start_time, 1)

			print(f'{len(text_generations)} sample(s) of text generated in {duration} seconds.')

			return max(text_generations, key=len)

		except Exception as e:
			logger.error(f":: An error has occurred while attempting to generate text")
			logger.error(e)
			raise e
		finally:
			pass
