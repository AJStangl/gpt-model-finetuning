Thread Conversation Language Modeling Fine-Tuning and Generation
===

Getting Started
---

This is a simple command line/library for generating a custom gtp2 fine-tuning data and subsequent fine-tuning.

Installation
---

The module can be directly installed through cloning the repository and installing via pip:

```bash
pip install  git+https://github.com/ajstangl/gpt-model-finetuning@master
``` 

Or you can clone the repository and set it up like so:

```bash
git clone https://github.com/AJStangl/gpt-model-finetuning.git
cd gpt-model-finetuning
python setup.py build
pip install --editable .
```

    Installing collected packages: gpt2-fine-tuning
    Running setup.py develop for gpt2-fine-tuning
    Successfully installed gpt2-fine-tuning-0.0.1


Usage
---

The module can be used in two ways. As a command line utility of as package in src.

CLI Usage
___

```bash
gpt2-fine-tuning --help
```

    Options:
      --help  Show this message and exit.
    
    Commands:
      download-data  Command for downloading fine-tuning data
      gen-text       command for generating text from a model
      train-model    Command for running fine-tuning for gpt-2 model


