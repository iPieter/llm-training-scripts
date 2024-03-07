import datasets
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
    AutoModelForCausalLM, AutoTokenizer
import os

os.environ["WANDB_PROJECT"]="test_llm"


def pack(dataset, tokenizer, context_length, key='text'):
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
    for row in dataset:
        ids = tokenizer(row[key], max_length=None)['input_ids']

        # end-of-sentence-token seems to have been present in Mistral 7B training data,
        # but is not automatically added by the tokenizer
        ids.append(2)

        cache.extend(ids)
        while len(cache) >= context_length:
            chunk = cache[:context_length]
            yield {'input_ids': chunk,
                   'attention_mask': [1] * context_length,
                   'labels': chunk}
            cache = cache[context_length:]

def train(base_model, context_length, dataset_name, dataset_subname, new_model_name):
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=False,
                                                 attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    # fix padding (mostly for inference, later for finetuning changed to unk_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # data
    dataset = datasets.load_dataset(dataset_name, dataset_subname, streaming=True)

    # it is customary to train LLMs by fully "packing" the context length with
    # fragments of one or more documents
    packed_train_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={'dataset': dataset['train'],
                    'tokenizer': tokenizer,
                    'context_length': context_length})

    #packed_validation_dataset = datasets.IterableDataset.from_generator(
    #    generator=pack,
    #    gen_kwargs={'dataset': dataset['validation'],
    #                'tokenizer': tokenizer,
    #                'context_length': context_length})

    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8
    training_steps = 10_000_000_000 // (torch.cuda.device_count() * per_device_train_batch_size *
                                       gradient_accumulation_steps * context_length)

    save_steps = training_steps // (6 * 4) + 1
    eval_steps = training_steps // (6 * 8) + 1
    # training
    training_args = TrainingArguments(
        max_steps=training_steps,
        optim='adamw_bnb_8bit',
        learning_rate=2e-5,
        lr_scheduler_type='cosine',
        warmup_steps=int(training_steps * 0.1),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        #evaluation_strategy='steps',
        #eval_steps=eval_steps,
        #per_device_eval_batch_size=per_device_train_batch_size,
        #eval_accumulation_steps=gradient_accumulation_steps,
        save_strategy='steps',
        save_steps=save_steps,
        bf16=True,
        output_dir='/tmp/geitje/output',
        report_to=['wandb'],
        logging_steps=1,
        logging_first_step=True,
        hub_model_id=new_model_name,
        hub_private_repo=True,
        push_to_hub=False,
        hub_strategy='all_checkpoints',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=packed_train_dataset,
        #eval_dataset=packed_validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()


if __name__ == '__main__':
    train(
        base_model='mistralai/Mistral-7B-v0.1',
        context_length=8192,
        dataset_name='oscar',
        dataset_subname='unshuffled_deduplicated_nl',
        new_model_name='Rijgersberg/GEITje-7B',
    )