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
    Phi3Config,
)
import os
import requests
from huggingface_hub import configure_http_backend, get_session
from tokenizers import AddedToken
import signal
import torch

from modeling_phi3 import (
    Phi3ForCausalLM,
)


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
    tokens = {y : tokenizer.get_vocab()[f'<s_{y}>'] for y in ["2014", "2016", "2018", "2020", "2022", "2024"]}

    for row in dataset:
        clean_text = re.sub(r'File:[^|]+\|','',re.sub(r'(\s+\t)', ' | ', row[key]))
        ids = tokenizer(clean_text, max_length=None)["input_ids"]
        ids[0] = tokens[row['year']]

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


def extend_tokenizer(tokenizer):
    # add special tokens for the years
    tokens = [
        AddedToken(f"<s_{year}>", single_word=True, lstrip=True, rstrip=True)
        for year in ["2014", "2016", "2018", "2020", "2022", "2024"]
    ]
    tokenizer.add_tokens(tokens, special_tokens=True)

    print("Added the following tokens:")
    print(tokens)
    return tokenizer


def train(
    base_model, context_length, dataset_name, dataset_subname, new_model_name, args
):

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer = extend_tokenizer(tokenizer)

    config = Phi3Config.from_pretrained(base_model)
    config._attn_implementation = "sdpa"
    config.output_attentions = False
    config.torch_dtype = "bfloat16"
    model = Phi3ForCausalLM(
        config #, torch_dtype=torch.bfloat16, attn_implementation="sdpa", output_attentions=False
    )  # pytorch flash attn implementation
    model.resize_token_embeddings(len(tokenizer))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated model size: {total_params * 2 / 1024**3:.2f} GB")  # Assuming bfloat16
    
    print(f"Model supports sdpa: {model._supports_sdpa}")
    # fix padding (mostly for inference, later for finetuning changed to unk_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # data
    dataset = datasets.load_dataset(
        dataset_name,
        dataset_subname,
        streaming=True,
        trust_remote_code=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    dataset = dataset.shuffle(seed=43, buffer_size=20_000)

    # it is customary to train LLMs by fully "packing" the context length with
    # fragments of one or more documents
    packed_train_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={
            "dataset": dataset["train"],
            "tokenizer": tokenizer,
            "context_length": context_length,
        },
    )

    packed_validation_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={
            "dataset": dataset["validation"],
            "tokenizer": tokenizer,
            "context_length": context_length,
        },
    )

    per_device_train_batch_size = 4
    gradient_accumulation_steps = 8
    training_steps = 10_000_000_000 // (
        torch.cuda.device_count()
        * per_device_train_batch_size
        * gradient_accumulation_steps
        * context_length
    )

    print(f"Total tokens per training step: {per_device_train_batch_size * gradient_accumulation_steps * context_length * torch.cuda.device_count()}")

    save_steps = 100 #training_steps // (6 * 4) + 1
    eval_steps = 200 #training_steps // (6 * 2) + 1
    
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
        gradient_checkpointing_kwargs={'use_reentrant':False},
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
        eval_dataset=packed_validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # overwrite lr scheduler
    # trainer.create_scheduler(num_training_steps=training_steps)
    
    signal.signal(signal.SIGINT, lambda signal_number, frame: save_checkpoint_and_exit(trainer))

    trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    # Parse cli args
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Location of the cache directory (for HF datasets etc)",
    )
    parser.add_argument("--proxy",  action='store_true', help="Whether to use a proxy")

    args = parser.parse_args()

    if args.proxy:
        # Set it as the default session factory
        print("using proxy")
        configure_http_backend(backend_factory=backend_factory)

    train(
        base_model="microsoft/Phi-3-mini-4k-instruct",
        #base_model="mistralai/Mistral-7B-v0.1",
        context_length=4096,
        dataset_name="pdelobelle/enwiki-yearly-cleaned",
        dataset_subname="full",
        new_model_name="pdelobelle/wikiPT-2B-2024",
        args=args,
    )
