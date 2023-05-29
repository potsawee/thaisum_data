import os
import argparse
import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(
    translation_model, # e.g. facebook/nllb-200-3.3B
    dataset,           # e.g. xsum
    data_split,        # e.g. train, validation, test
    data_type,         # e.g. summary, article
    max_length,        # default 1024
    cache_batch_size,  # default 1000
    output_dir,
    shuffle,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(translation_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(translation_model)
    model = model.eval()
    model = model.to(device)
    print("loaded:", translation_model)

    dataset = load_dataset(dataset)[data_split]
    ids = [i for i in range(len(dataset))]
    id_chunks = list(chunks(ids, cache_batch_size))
    num_chunks = len(id_chunks)

    if shuffle:
        random.shuffle(id_chunks)

    for iter, id_chunk in enumerate(id_chunks):
        outpath = "{}/{}_{}.json".format(output_dir, id_chunk[0], id_chunk[-1])
        exist = os.path.isfile(outpath)
        if exist:
            print("id {}: already exists".format(idx))
            continue
        translated_texts = {}
        for idx in id_chunk:
            source = dataset[idx][data_type]
            bbcid = dataset[idx]['id'] # quick hack

            inputs = tokenizer(source, return_tensors="pt").to(device)
            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["tha_Thai"], max_length=max_length,
            )
            translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_texts[idx] = {'text': translated_text, 'id': bbcid}

        with open(outpath, 'w') as f:
            json.dump(translated_texts, f)
        print(f"iter{iter}/{num_chunks}, outpath={outpath}")

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--translation_model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_split', type=str, required=True)
    parser.add_argument('--data_type', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--cache_batch_size', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shuffle', type="bool", nargs="?", const=True, default=False)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    with torch.no_grad():
        main(**kwargs)
