import logging
from src.training_module.model_fine_tuning import ModelFineTuner

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	bot_label = "only_fans_3"
	# data = "training.csv"
	model_fine_tuner = ModelFineTuner(bot_label)

	# train_file, eval_file = model_fine_tuner.generate_text_training_data(data, bot_label, ["onlyfansadvice", "CreatorsAdvice"])

	model = model_fine_tuner.get_gpt2_model()

	model_fine_tuner.train_model(model, train_file="D:\\code\\repos\\quick_training\\only_fans_3_train.txt", eval_file="D:\\code\\repos\\quick_training\\only_fans_3_eval.txt")