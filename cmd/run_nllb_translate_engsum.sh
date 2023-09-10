#!/bin/bash
#$ -S /bin/bash
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped

export START=$1
source ~/anaconda3/bin/activate py38-2022

export TRANSFORMERS_CACHE='/home/alta/summary/pm574/.cache/models'
export HF_DATASETS_CACHE='/home/alta/summary/pm574/.cache/datasets'
export PYTHONBIN='/home/mifs/pm574/anaconda3/envs/py38-2022/bin/python'

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

$PYTHONBIN /home/alta/summary/pm574/thainlp/thaisum_data/nllb_translate_engsum.py \
  --translation_model facebook/nllb-200-3.3B \
  --dataset xsum \
  --data_split train \
  --data_type summary \
  --max_length 512 \
  --cache_batch_size 1000 \
  --output_dir /home/alta/summary/pm574/thainlp/thaisum_data/translated_texts/xsum/summary/train \
  --shuffle True

# qsub -cwd -j yes -o LOGs/nllb_translate_engsum_0.txt -P esol -l qp=cuda-low -l gpuclass='volta' -l osrel='*' ./cmd/run_nllb_translate_engsum.sh
# -l gpuclass='volta' -l hostname=air213.eng.cam.ac.uk
