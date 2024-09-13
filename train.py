import argparse
import re
import datasets
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    SchedulerType,
)
import os
import requests
from huggingface_hub import configure_http_backend, get_session
from tokenizers import AddedToken
import signal
import torch

from modeling_gemma2 import MultiheadGemma2ForCausalLM, eu_languages

from datasets import concatenate_datasets, interleave_datasets, load_dataset

desired_datasets = [
    ("legal_mc4", 1),
    ("open_discourse_bundestag", 2),
    ("opensubtitles", 1),
    ("oscar_2015_14", 1),
    ("oscar_2016_40", 1),
    ("oscar_2017_43", 1),
    ("oscar_2018_47", 1),
    ("oscar_2019_22", 1),
    ("oscar_2020_24", 1),
    ("oscar_2020_45", 1),
    ("oscar_2021_49", 1),
    ("oscar_2022_27", 1),
    #("oscar_2022_49", 1),
    ("oscar_2023_14", 1),
    ("oscar_2023_23", 1),
    ("eurlex", 1),
    ("parlamint", 1),
    ("tagesschau_2018_2023", 2),
    ("wikibooks", 2),
    ("wikinews", 2),
    ("wikipedia_euro", 4),
    ("wikiquote", 2),
    ("wikisource", 1),
    ("wikivoyage", 1),
]

# Create a factory function that returns a Session with configured proxies
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.proxies = {"http": "http://proxy:80", "https": "http://proxy:80"}
    return session


def save_checkpoint_and_exit(trainer):
    # Save the state
    trainer.save_state()
    print(f"State saved to {trainer.args.output_dir}. Exiting.")
    exit(0)


def pack(dataset, tokenizer, context_length, key="text"):
    """Concatenate ("pack") samples from a dataset into tokenized chunks of `context_length`.

    Used for efficient training of causal models without padding. No special measures are taken
    to disallow a sequence attending to a previous sequence. The model is left to learn the
    unrelatedness of sequences from the presence of the start- and end-of-sequence-tokens
    between the samples, following a similar convention from GPT-3 and T5.
    See https://github.com/huggingface/transformers/issues/17726 for a feature request for
    Hugging Face Transformers.

    The incomplete final chunk is discarded.

    :param dataset: Dataset of samples (iterable of dict-like, e.g. Hugging Face dataset)
    :param tokenizer: Callable that tokenizes the samples (e.g. Hugging Face tokenizer)
    :param context_length: number of tokens in packed sequences
    :param key: key of the text field in the sample. Defaults to 'text'
    :yield: dicts of packed input_ids, attention_masks and (self-supervised) labels
    """
    cache = []
    #tokens = {y: tokenizer.get_vocab()[f"<s_{y}>"] for y in ["2014", "2016", "2018", "2020", "2022", "2024"]}

    for row in dataset:
        clean_text = re.sub(r"File:[^|]+\|", "", re.sub(r"(\s+\t)", " | ", re.sub(r"\s+", " ", row[key])))
        ids = tokenizer(clean_text, max_length=None)["input_ids"]
        #ids[0] = tokens[row["year"]]

        # end-of-sentence-token seems to have been present in Mistral 7B training data,
        # but is not automatically added by the tokenizer
        # ids.append(2)

        cache.extend(ids)
        while len(cache) >= context_length:
            chunk = cache[:context_length]
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * context_length,
                "labels": chunk,
            }
            cache = cache[context_length:]


def train(base_model, context_length, dataset_name, dataset_subname, new_model_name, args):

    tokenizers = { lang: AutoTokenizer.from_pretrained(tokenizer, use_fast=False) for _, lang, _, _, tokenizer in eu_languages}
    
    tokenizer = tokenizers['de']

    model = AutoModelForCausalLM.from_pretrained(base_model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    model.config._attn_implementation = "sdpa"
    model.config.output_attentions = False
    model.config.torch_dtype = "bfloat16"
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # saving some space
    del model.model.embed_tokens['en']
    del model.lm_head['en']
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters after trimming 'en' weights: {total_params:,}")
    print(f"Estimated model size: {total_params * 2 / 1024**3:.2f} GB")  # Assuming bfloat16

    print(f"Model supports sdpa: {model._supports_sdpa}")
    # fix padding (mostly for inference, later for finetuning changed to unk_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # model.config.pad_token_id = model.config.eos_token_id

    # data
    dataset = interleave_datasets([
        load_dataset('occiglot/occiglot-fineweb-v0.5', data_dir=f"de/{d}", streaming=True)[dataset_subname] for d, _ in desired_datasets],
        probabilities=[p/(sum([pp for _, pp in desired_datasets])) for _, p in desired_datasets],
        stopping_strategy="all_exhausted"
    )

    dataset = dataset.shuffle(seed=43)

    # it is customary to train LLMs by fully "packing" the context length with
    # fragments of one or more documents
    packed_train_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={
            "dataset": dataset,
            "tokenizer": tokenizer,
            "context_length": context_length,
        },
    )

    dataset_test = load_dataset("wikimedia/wikipedia", "20231101.de", split="train", streaming=True)
    dataset_test = dataset_test.take(2000)

    dataset_toxic = load_dataset("textdetox/multilingual_toxicity_dataset", split="de", streaming=True)
    dataset_toxic = dataset_toxic.take(500)

    packed_wiki_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={
            "dataset": dataset_test,
            "tokenizer": tokenizer,
            "context_length": context_length,
        },
    )

    packed_toxic_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={
            "dataset": dataset_toxic,
            "tokenizer": tokenizer,
            "context_length": context_length,
        },
    )

    per_device_train_batch_size = 4
    gradient_accumulation_steps = 8
    training_steps = 10_000_000_000 // (
        torch.cuda.device_count() * per_device_train_batch_size * gradient_accumulation_steps * context_length
    )

    print(
        f"Total tokens per training step: {per_device_train_batch_size * gradient_accumulation_steps * context_length * torch.cuda.device_count()}"
    )

    save_steps = 300  # training_steps // (6 * 4) + 1
    eval_steps = 300  # training_steps // (6 * 2) + 1

    # training
    training_args = TrainingArguments(
        max_steps=training_steps,
        optim="adamw_bnb_8bit",
        learning_rate=2e-4,
        lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
        weight_decay=0.01,
        adam_beta2=0.95,
        warmup_steps=int(training_steps * 0.05),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        do_eval=False,
        eval_strategy="steps",
        eval_steps=eval_steps,
        per_device_eval_batch_size=per_device_train_batch_size,
        save_strategy="steps",
        include_num_input_tokens_seen=True,
        save_steps=save_steps,
        bf16=True,
        ignore_data_skip=True,
        output_dir=args.output_dir,
        report_to=["wandb"],
        logging_steps=1,
        logging_first_step=True,
        hub_model_id=new_model_name,
        hub_private_repo=True,
        push_to_hub=True,
        hub_strategy="all_checkpoints",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=packed_train_dataset,
        eval_dataset={"wikipedia": packed_wiki_dataset, "toxicity": packed_toxic_dataset},
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # overwrite lr scheduler
    # trainer.create_scheduler(num_training_steps=training_steps)

    signal.signal(signal.SIGINT, lambda signal_number, frame: save_checkpoint_and_exit(trainer))

    trainer.train()


if __name__ == "__main__":
    # Parse cli args
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--output-dir", type=str, help="Output directory", default="/tmp/llm/")
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Location of the cache directory (for HF datasets etc)",
    )
    parser.add_argument("--proxy", action="store_true", help="Whether to use a proxy")

    args = parser.parse_args()

    if args.proxy:
        # Set it as the default session factory
        print("using proxy")
        configure_http_backend(backend_factory=backend_factory)

    train(
        base_model="pdelobelle/gemma-2-2b-de",
        # base_model="mistralai/Mistral-7B-v0.1",
        context_length=8192,
        dataset_name="occiglot/occiglot-fineweb-v0.5",
        dataset_subname="train",
        new_model_name="pdelobelle/gemma-2-2b-multihead-de",
        args=args,
    )
