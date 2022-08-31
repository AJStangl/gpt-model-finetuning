import os

from transformers import GPT2Tokenizer, GPT2LMHeadModel
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
	tokenizer = GPT2Tokenizer.from_pretrained('./results/tokenizer')
	model = GPT2LMHeadModel.from_pretrained('D:\\code\\repos\\quick_training\\results\\checkpoint-1000').cuda()
	foo = "<|soss r/onlyHams|><|sot|>Stop having issues<|eot|><|sost|>Just stop it.<|eost|><|soopr u/arzen221|>the tests will continue until the problem is corrected.<|eoopr|><|soopr|>"
	generated = tokenizer(f"<|startoftext|> {foo}", return_tensors="pt")

	sample_outputs = model.generate(inputs=generated.input_ids.cuda(), attention_mask=generated['attention_mask'].cuda(), do_sample=True, top_k=40, max_length=1024, top_p=0.8, temperature=0.8, num_return_sequences=1, repetition_penalty=1.08, stop_token='<|endoftext|>')

	for i, sample_output in enumerate(sample_outputs):
		result = tokenizer.decode(sample_output, skip_special_tokens=True)
		print("{}: {}".format(i, result.replace(foo, "")))