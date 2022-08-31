import pandas
import csv

from src.text_generation.model_text_generator import ModelTextGenerator


def main(subreddits: [str]):
	model_path: str = "D:\\code\\repos\\quick_training\\results\\checkpoint-500\\"
	text_generator: ModelTextGenerator = ModelTextGenerator()
	foo = text_generator.generate_text_with_no_wrapper(model_path, "<|soss r/onlyHams|><|sot|>Stop having issues<|eot|><|sost|>Just stop it.<|eost|><|soopr u/arzen221|>the tests will continue until the problem is corrected.<|eoopr|><|soopr|>")


# train_text_file = 'f_train.txt'
# eval_text_file = 'f_eval.txt'
# subreddits = ["onlyfansadvice", "CreatorsAdvice"]
# df = pandas.read_csv("training.csv")
# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# # Drop unnamed columns
# df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
# # Clear duplicates
# df = df.T.drop_duplicates().T
#
# filtered = df.where(df['CommentBody'] != '[deleted]')
# filtered.dropna(inplace=True, subset=['Subreddit'])
#
# filtered = filtered.where(filtered['CommentBody'] != '[removed]')
# filtered.dropna(inplace=True, subset=['Subreddit'])
#
# filtered = filtered.where(filtered['Subreddit'].isin(subreddits))
# filtered.dropna(inplace=True, subset=['Subreddit'])
#
# train_df = filtered.sample(frac=.9)
# train_ids = list(train_df['CommentId'])
#
# eval_df = filtered.where(~filtered['CommentId'].isin(train_ids))
# eval_df.dropna(inplace=True, subset=['Subreddit'])
#
# training_text = train_df["TrainingString"].apply(lambda x: x.replace('\n', '\\n'))
#
# eval_text = eval_df["TrainingString"].apply(lambda x: x.replace('\n', '\\n'))
#
# training_text.to_csv(train_text_file, index=False, header=False, line_terminator='\n', encoding="utf-8",
# 					 escapechar='\\', quoting=csv.QUOTE_NONE)
# eval_text.to_csv(eval_text_file, index=False, header=False, line_terminator='\n', encoding="utf-8", escapechar='\\',
# 				 quoting=csv.QUOTE_NONE)

if __name__ == '__main__':
	main(["onlyfansadvice", "CreatorsAdvice"])
