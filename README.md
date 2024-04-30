# SMART-LLaVa

## SMART Dataset
Download the dataset using:

```bash
wget https://zenodo.org/records/7775984/files/SMART101-release-v1.zip
unzip SMART101-release-v1.zip
```

## Zero-Shot Model Baselines

The Model_baselines.ipynb file can be run to evaluate each of the models on the test data. These are the zero-shot generalization benchmarks.

## LLaVa model prompt tuning

The LLaVa_baselines.py file prompt tunes a LLaVa/BakLLaVa model on SMART data. The prompt tokens are injected between the image and language instruction tokens.

<p align="center">
<img src="https://github.com/akankshya107/SMART-LLaVa/blob/main/trainable_prompt.png" width="500">
</p>

The code trains and uploads a model to huggingface. The huggingface token must be provided as $HUGGINGFACE_TOKEN before running the code. The following packages need to be installed before running the code.

```
pip install transformers
pip install peft
pip install bitsandbytes==0.41.3 accelerate==0.25.0
python3 LLaVa_baselines.py
```

## Evaluating puzzle types
The relevant files are LLaVa_types.ipynb and resnet.py
