from typing import Literal
import json
import yaml
import ast
from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
import functools
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

def run_finetune(model_name:str,dataset_name:Literal['bfcl','xlam'],json_or_yaml:Literal['json','yaml']):
    
    
    dtype = torch.bfloat16 
    load_in_4bit = False 

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # or choose "unsloth/Llama-3.2-1B"
        # max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        trust_remote_code=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 64,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    if dataset_name == "bfcl":
        train_data = []
        with open('train.json', 'r') as file:
            for line in file:
                train_data.append(json.loads(line.strip()))
        with open('test.json', 'r') as file:
            test_data = json.load(file)
        # train_data = _process_bfcl_train_data(train_data)
    
        train_ds = Dataset.from_list(train_data)
        train_ds = train_ds.map(functools.partial(_get_bfcl_tokenized_ds,tokenizer=tokenizer,json_or_yaml=json_or_yaml),batched=True)
    
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    dataset_text_field = "prompt",
    max_seq_length = 3072,
    dataset_num_proc = 8,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        # warmup_steps = 5,
        num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = f"outputs_{model_name}_{json_or_yaml}_{dataset_name}",
        save_safetensors=True
    ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>system<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    trainer_stats = trainer.train()
    return trainer_stats, trainer, model, tokenizer
    

def _get_bfcl_tokenized_ds(examples,tokenizer, json_or_yaml: Literal["json","yaml"]):
    user_prompts = examples['Instruction']
    functions = examples['Functions']
    outputs = examples['Output']
    prompts = []
    for up,fn, op in zip(user_prompts,functions, outputs):
        if json_or_yaml == "json":
            fn = json.dumps(fn,indent=1)
        elif json_or_yaml == "yaml":
            fn = json_to_yaml(fn)
        prompts.append(tokenizer.apply_chat_template(
            _create_messages(up,fn, op),
            tokenize=False
        ))

    return {"prompt":prompts,}
    
def json_to_yaml(data):
    curr_func_yaml = ""
    json_func = ast.literal_eval(data)
    for func in json_func:
        curr_func_yaml+=yaml.dump(ast.literal_eval(func)) + "\n\n"
    return curr_func_yaml
    
def _process_bfcl_train_data(train_data):
    for td in train_data:
        output = td['Output']
        td['Functions'] = str(td['Functions'])
        td['yaml_function'] = json_to_yaml(td['Functions'])
        td['json_function'] = td['Functions'][1:-1]
        del td['Functions']
        
        if isinstance(output,str):
            pass
        elif isinstance(output,list):
            output = output[0]
        td['Output'] = output
    return train_data

def process_ast_node(node):
    # Check if the node is a function call
    if isinstance(node, ast.Call):
        # Return a string representation of the function call
        return ast.unparse(node) 
    else:
        # Convert the node to source code and evaluate to get the value
        node_str = ast.unparse(node)
        return eval(node_str)

      
def parse_python_function_call(call_str):
    tree = ast.parse(call_str)
    expr = tree.body[0]

    call_node = expr.value
    function_name = (
        call_node.func.id
        if isinstance(call_node.func, ast.Name)
        else str(call_node.func)
    )

    parameters = {}
    noNameParam = []

    # Process positional arguments
    for arg in call_node.args:
        noNameParam.append(process_ast_node(arg))

    # Process keyword arguments
    for kw in call_node.keywords:
        parameters[kw.arg] = process_ast_node(kw.value)

    if noNameParam:
        parameters["None"] = noNameParam
        
    function_dict = {"name": function_name, "arguments": parameters}
    return function_dict

def _create_messages(user_prompt:str,functions:str, output:str):
    messages = [
            {
                "role": "system",
                "content": "You are an expert in composing functions. You are given a question and a set of possible functions."
                            " Based on the question, you will need to make one or more function/tool calls to achieve the purpose."
                            " If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections.\n",
            },
            {"role": "user", "content": f"#### Question: {user_prompt}Here is a list of functions that you can invoke:\n{functions}. Should you decide to return the function call(s), NO other text MUST be included.\n#### Response:"},
            {"role":"assistant","content":output}
        ]
    return messages