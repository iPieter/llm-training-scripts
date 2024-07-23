import sys
import torch
from datasets import DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def train(model, tokenizer, chat_dataset, new_model_name):

    def format(examples):
        return [tokenizer.apply_chat_template(conversation, tokenize=False)
                for conversation in examples['messages_nl']]

    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    steps_per_epoch = len(chat_dataset['train_sft'])\
                 // (torch.cuda.device_count() * per_device_train_batch_size * gradient_accumulation_steps)
    eval_steps = steps_per_epoch // 5

    training_args = TrainingArguments(
        optim='adamw_bnb_8bit',
        num_train_epochs=3,
        learning_rate=1e-5,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        evaluation_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='epoch',
        bf16=True,
        output_dir='/scratch/leuven/328/vsc32851/llm-output-sft/',
        report_to=['wandb'],
        logging_steps=1,
        logging_first_step=True,
        hub_model_id=new_model_name,
        push_to_hub=True,
        hub_private_repo=True,
        hub_strategy='all_checkpoints',
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=8192,
        train_dataset=chat_dataset['train_sft'],
        eval_dataset=chat_dataset['test_sft'],
        formatting_func=format,
        neftune_noise_alpha=5,
    )

    trainer.train()


if __name__ == '__main__':
    basemodel_name = sys.argsv[1]
    model = AutoModelForCausalLM.from_pretrained(basemodel_name, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, use_flash_attention_2=True,
                                                 device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(basemodel_name)

    tokenizer.pad_token = tokenizer.pad_token
    model.config.pad_token_id = tokenizer.unk_token_id

    no_robots_nl = load_dataset('Rijgersberg/no_robots_nl')
    ultrachat_nl = load_dataset('Rijgersberg/ultrachat_10k_nl')

    chat_dataset = DatasetDict({
        'train_sft': concatenate_datasets([no_robots_nl['train_sft'],
                                           ultrachat_nl['train_sft']]).shuffle(seed=42),
        'test_sft': concatenate_datasets([no_robots_nl['test_sft'],
                                          ultrachat_nl['test_sft']]).shuffle(seed=42),
    })

    chat_dataset = chat_dataset.filter(lambda row: all(turn['content'] != '<TRANSLATION FAILED>'
                                                       for turn in row['messages_nl']))

    train(model, tokenizer, chat_dataset,
          new_model_name='pdelobelle/tweety-7b-chat-v24a')