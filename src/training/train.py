from unsloth import FastLanguageModel
import torch
import json
from trl import SFTConfig, SFTTrainer
from data import prepare_data
import argparse

def main(args):
    max_seq_length = 2048 
    dtype = None 
    load_in_4bit = True 

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,  
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 32,    
        lora_dropout = 0.05, 
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    dataset = prepare_data(train_path=args.train_path)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,

            num_train_epochs = 2,       
            warmup_ratio = 0.03,         

            learning_rate = 2e-4,
            logging_steps = 10,

            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",

            seed = 3407,
            output_dir = "outputs",
            report_to = "none",         
        ))

    trainer.train()

    trainer.model.save_pretrained(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)
    print(f"✅ LoRA fine-tuned model saved to {args.save_model_dir}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, )
    parser.add_argument("--train_path", type=str, required=True, )
    parser.add_argument("--save_model_dir",type=str,)

    args = parser.parse_args()
    main(args)
