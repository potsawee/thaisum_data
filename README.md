# ThaiSum Data
- This repository contains the code to generate (i.e., translate) standard Summarization datasets from English to Thai
- Currently, the translation system is *Meta's NLLB-200*, but other systems can be considered too
- Source = Input Document, Target = Gold Summary
- The translated outputs can be used for experimenting with the cascaded approach or training any models

## Translated Datasets
The translated datasets are open-sourced on HuggingFace, so you can simply download them, for example,

```python
from datasets import load_dataset
dataset = load_dataset("potsawee/xsum_thai")
```

Completed translated datasets include:

|        Dataset       |  Source |  Target | Size (train/val/test) | Link                                                        |
|:--------------------:|:-------:|:-------:|:---------------------:|-------------------------------------------------------------|
|    `xsum_eng2thai`   | &cross; | &check; |   204045/11332/11334  | https://huggingface.co/datasets/potsawee/xsum_eng2thai      |
|      `xsum_thai`     | &check; | &check; |   204045/11332/11334  | https://huggingface.co/datasets/potsawee/xsum_thai          |
| `cnn_dailymail_thai` | &check; | &check; |   287113/13368/11490  | https://huggingface.co/datasets/potsawee/cnn_dailymail_thai |

## To-do
- [ ] Check and add License for each dataset
- [ ] Add data statistics, e.g., length, n-gram overlap etc
- [ ] Train (fine-tune) baseline summarization models
