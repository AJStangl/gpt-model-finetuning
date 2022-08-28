from setuptools import setup
import os


def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name="gpt2-fine-tuning",
	version="0.0.1",
	author="AJ Stangl",
	author_email="ajstangl@gmail.com",
	description="A simple wrapper for collecting, transforming, and creating gpt2 models based on reddit subreddits",
	license="MIT",
	keywords="GPT2",
	include_package_data=True,
	url="https://example.com",
	packages=['src', 'src/reddit', 'src/training_module', 'src/training_module/data_model', 'src/tagging'],
	long_description=read('README.md'),
	classifiers=[
		"Topic :: Utilities",
		"License :: MIT License",
	],
	entry_points={
		'console_scripts': [
			'gpt2-fine-tuning = src.cli_wrapper:cli',
		],
	},
)
