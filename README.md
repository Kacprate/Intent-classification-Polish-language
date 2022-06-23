# Intent-classification-Polish-language
A deep learning solution for intent classification problem in the domain of Polish language.

- HerBERT model on HuggingFace: https://huggingface.co/allegro

#### Approach
Intent classification is done by using a HerBERT as a backbone language model to understand the utterances. On top of that, a custom intent classification head is used to output the utterance intent.

#### Dataset
MASSIVE: https://arxiv.org/pdf/2204.08582.pdf

#### Useful files
- `train.py` - train code
- `test.py` - test code with csv export for further usage in the `analytics.py`
- `analytics.py` - code that extracts interesting information about the trained model, it uses the csv file exported using `test.py` code
