#!/bin/bash
#$ -S /bin/bash

export START=$1

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh" ]; then
        . "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source activate py38-2023

# huggingface datasets lib
export PYTORCH_TRANSFORMERS_CACHE=/home/pm574/rds/rds-altaslp-8YSp2LXTlkY/data/cache/huggingface/transformers
export HF_DATASETS_CACHE=/home/pm574/rds/hpc-work/downloads/hf_datasets
export PYTHONBIN=/home/pm574/.conda/envs/py38-2023/bin/python

$PYTHONBIN /home/pm574/rds/hpc-work/thainlp/thaisum_data/nllb_translate_engsum.py \
  --translation_model facebook/nllb-200-3.3B \
  --dataset xsum \
  --data_split train \
  --data_type document \
  --max_length 1024 \
  --cache_batch_size 100 \
  --output_dir /home/pm574/rds/hpc-work/thainlp/thaisum_data/translated_texts/xsum/document_v2/train \
  --shuffle True
