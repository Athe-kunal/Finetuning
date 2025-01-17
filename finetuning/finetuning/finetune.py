from typing import Literal
import json
import yaml
import ast
from unsloth import FastLanguageModel
import unsloth
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset, DatasetDict
import functools
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from typing import NamedTuple
from tqdm import tqdm
import wandb
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from transformers.trainer_callback import TrainerCallback, TrainerState
import fire

load_dotenv(find_dotenv(), override=True)
wandb.login(key=os.getenv("WANDB_API_KEY"))


class ModelReturn(NamedTuple):
    model_answer: str
    gt_answer: str
    score: float


class PeftSavingCallback(TrainerCallback):
    output_dir: str

    def on_epoch_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(self.output_dir, f"epoch_{int(state.epoch)}")
        kwargs["model"].save_pretrained(peft_model_path)


def run_finetune(
    model_name: str,
    dataset_name: Literal["bfcl", "xlam"],
    json_or_yaml: Literal["json", "yaml"],
) -> None:
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        group=f"{model_name}_{json_or_yaml}_{dataset_name}",
    )

    dtype = torch.bfloat16
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # or choose "unsloth/Llama-3.2-1B"
        # max_seq_length = max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    chat_template = "mistral" if "mistral" in model_name else "llama-3.1"
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )
    if dataset_name == "bfcl":
        train_data = []
        with open("train.json", "r") as file:
            for line in file:
                json_obj = json.loads(line.strip())
                json_obj["Functions"] = (
                    json_obj["Functions"][0]
                    if isinstance(json_obj["Functions"], list)
                    else json_obj["Functions"]
                )
                json_obj["Output"] = (
                    json_obj["Output"][0]
                    if isinstance(json_obj["Output"], list)
                    else json_obj["Output"]
                )
                train_data.append(json_obj)
        with open("test.json", "r") as file:
            test_data = json.load(file)
        train_ds = Dataset.from_list(train_data)
        train_ds = train_ds.map(
            functools.partial(
                _get_bfcl_train_tokenized_ds,
                tokenizer=tokenizer,
                json_or_yaml=json_or_yaml,
            ),
            batched=True,
        )
        test_ds = Dataset.from_list(test_data)
        test_ds = test_ds.map(
            functools.partial(
                _get_bfcl_tokenized_test_ds,
                tokenizer=tokenizer,
                json_or_yaml=json_or_yaml,
            ),
            batched=True,
            remove_columns=["function", "question"],
        )
    elif dataset_name == "xlam":
        ds = DatasetDict.load_from_disk(f"xlam_data_{json_or_yaml}")["train"]
        split_ds = ds.train_test_split(test_size=1000, seed=42)
        train_ds = split_ds["train"]
    output_dir = f"outputs_{model_name}_{json_or_yaml}_{dataset_name}"
    peft_callback = PeftSavingCallback()
    peft_callback.output_dir = output_dir
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="prompt",
        max_seq_length=3072,
        dataset_num_proc=8,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,  # Can make training 5x faster for short sequences.
        callbacks=[peft_callback],
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            do_eval=False,
            # warmup_steps = 5,
            num_train_epochs=3,  # Set this for 1 full training run.
            # max_steps=1,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_safetensors=True,
            report_to="wandb",
            # save_strategy="epoch",
            run_name=f"{model_name}_{json_or_yaml}_{dataset_name}",
        ),
    )
    if chat_template == "mistral":
        pass
    else:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>system<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    trainer_stats = trainer.train()


def run_evaluation(
    model_name: str,
    model_path: str,
    dataset_name: Literal["bfcl", "xlam"],
    json_or_yaml: Literal["json", "yaml"],
    val_batch_size: int = 32,
) -> None:
    child_path = model_path.split("/")[-1]
    assert child_path.startswith("epoch")
    dtype = torch.bfloat16
    run = wandb.init(
        project="JSON vs YAML Finetuning Project",
        entity="athe_kunal",
        group=f"{model_name}_{json_or_yaml}_{dataset_name}",
        name=f"{model_name}_{json_or_yaml}_{dataset_name}_{child_path}",
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path, dtype=dtype, load_in_4bit=False
    )
    tokenizer.padding_side = "left"
    tokenizer.chat_template = unsloth.chat_templates.llama31_template
    FastLanguageModel.for_inference(model)
    if dataset_name == "bfcl":
        with open("test.json", "r") as file:
            test_data = json.load(file)
        test_ds = Dataset.from_list(test_data)
        test_ds = test_ds.map(
            functools.partial(
                _get_bfcl_tokenized_test_ds,
                tokenizer=tokenizer,
                json_or_yaml=json_or_yaml,
            ),
            batched=True,
            remove_columns=["function", "question"],
        )
    elif dataset_name == "xlam":
        ds = DatasetDict.load_from_disk(f"xlam_data_{json_or_yaml}")["train"]
        split_ds = ds.train_test_split(test_size=1000, seed=42)
        test_ds = split_ds["test"]
        # test_ds = test_ds.select(range(10))
    scores_returned = evaluation_loop(
        test_ds, model, tokenizer, val_batch_size, dataset_name
    )
    table_data = [
        [score.model_answer, score.gt_answer, score.score] for score in scores_returned
    ]
    run.log(
        {
            "table_data": wandb.Table(
                data=table_data, columns=["Model Answer", "GT Answer", "Score"]
            )
        }
    )
    accuracy = sum([score.score for score in scores_returned]) / len(scores_returned)
    run.log({"accuracy": accuracy})
    run.finish()
    torch.cuda.empty_cache()


def evaluation_loop(
    test_ds: Dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_batch_size: int,
    dataset_name: Literal["bfcl", "xlam"],
) -> list[ModelReturn]:
    answer_col = "model_answer" if dataset_name == "bfcl" else "answers"
    scores: list[ModelReturn] = []
    for start in tqdm(range(0, len(test_ds), val_batch_size)):
        end = min(len(test_ds), start + val_batch_size)
        batch = test_ds[start:end]
        inputs = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda:0")

        model_outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            use_cache=True,
            do_sample=False,
        )
        str_outputs = tokenizer.batch_decode(model_outputs)
        for model_answer, gt_answer in zip(str_outputs, batch[answer_col]):
            if dataset_name == "xlam":
                gt_answer = _convert_answer(ast.literal_eval(gt_answer)[0])
            model_processed_answer = process_model_answer(model_answer)
            try:
                score = evaluation(model_processed_answer, gt_answer)
            except:
                score = 0.0
            if model_processed_answer == "":
                model_processed_answer = model_answer
            scores.append(
                ModelReturn(
                    model_answer=model_processed_answer,
                    gt_answer=gt_answer,
                    score=score,
                )
            )
    df = pd.DataFrame(
        [
            {"model_answer": s.model_answer, "gt_answer": s.gt_answer, "score": s.score}
            for s in scores
        ]
    )

    # Save DataFrame to CSV
    df.to_csv("evaluation_results.csv", index=False)
    return scores


def process_model_answer(out: str) -> str:
    out = out.replace("<|eot_id|>", "")
    out = out.replace("<|end_of_text|>", "")
    out = out.replace("<|eot_id|><|start_header_id|>assistant", "")
    out = out.replace("<|start_header_id|>assistant", "")
    start_idx = out.rindex("<|end_header_id|>") + len("<|end_header_id|>")
    return out[start_idx:].strip()


def evaluation(model_answer: str, gt_answer: str) -> float:
    model_answer = parse_python_function_call(model_answer)
    gt_answer = parse_python_function_call(gt_answer)

    if model_answer["name"] != gt_answer["name"]:
        return 0.0
    args_score = 0
    for model_answer_arg, model_answer_val in model_answer["arguments"].items():
        if (
            model_answer_arg not in gt_answer["arguments"]
            or gt_answer["arguments"][model_answer_arg] != model_answer_val
        ):
            args_score += 0
        else:
            args_score += 1
    return args_score / len(model_answer["arguments"])


def _get_bfcl_tokenized_test_ds(
    examples: dict, tokenizer: AutoTokenizer, json_or_yaml: Literal["json", "yaml"]
) -> dict:
    user_prompts = examples["question"]
    functions = examples["function"]
    prompts = []
    for up, fn in zip(user_prompts, functions):
        fn = fn[0] if isinstance(fn, list) else fn
        if json_or_yaml == "json":
            fn = json.dumps(fn, indent=1)
        elif json_or_yaml == "yaml":
            fn = json_to_yaml(f"[{fn}]")
        prompts.append(
            tokenizer.apply_chat_template(
                _create_messages(up, fn, ""), tokenize=False, add_generation_prompt=True
            )
        )
    return {
        "prompt": prompts,
    }


def _get_bfcl_train_tokenized_ds(
    examples: dict, tokenizer: AutoTokenizer, json_or_yaml: Literal["json", "yaml"]
) -> dict:
    user_prompts = examples["Instruction"]
    functions = examples["Functions"]
    outputs = examples["Output"]
    prompts = []
    for up, fn, op in zip(user_prompts, functions, outputs):
        if json_or_yaml == "json":
            fn = json.dumps(ast.literal_eval(fn), indent=1)
        elif json_or_yaml == "yaml":
            fn = json_to_yaml(f"[{fn}]")
        prompts.append(
            tokenizer.apply_chat_template(_create_messages(up, fn, op), tokenize=False)
        )

    return {
        "prompt": prompts,
    }


def process_ast_node(node):
    if isinstance(node, (ast.Constant, ast.Constant, ast.Constant)):
        return node.value
    elif isinstance(node, ast.List):
        return [process_ast_node(elt) for elt in node.elts]
    else:
        return ast.unparse(node)


def parse_python_function_call(call_str):
    tree = ast.parse(call_str)
    expr = tree.body[0].value

    def extract_function_name(node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{extract_function_name(node.value)}.{node.attr}"
        else:
            return ast.unparse(node)

    # return expr
    function_name = extract_function_name(expr.func)

    parameters = {}
    noNameParam = []

    # Process positional arguments
    for arg in expr.args:
        noNameParam.append(process_ast_node(arg))

    # Process keyword arguments
    for kw in expr.keywords:
        parameters[kw.arg] = process_ast_node(kw.value)

    if noNameParam:
        parameters["None"] = noNameParam

    function_dict = {"name": function_name, "arguments": parameters}
    return function_dict


def clean_string(s: str) -> str:
    """
    Cleans a string by encoding it to UTF-8 and replacing unencodable characters.
    """
    return s.encode("utf-8", errors="replace").decode("utf-8")


def remove_malinformed_str(data: str) -> str:
    data = data.replace("true", "True")
    data = data.replace("false", "False")
    data = data.replace("null", "None")
    return data


def process_xlam_data(
    example_list: dict, json_or_yaml: Literal["json", "yaml"], tokenizer: AutoTokenizer
) -> dict:
    prompts = []
    queries = example_list["query"]
    answers = example_list["answers"]
    tools = example_list["tools"]

    for i in range(len(queries)):
        try:
            try:
                # Clean the query string
                queries[i] = clean_string(queries[i])

                # Replace JSON literals with Python literals

                answers[i] = remove_malinformed_str(answers[i])
                tools[i] = remove_malinformed_str(tools[i])
                # Convert the first answer entry
                answer = _convert_answer(ast.literal_eval(answers[i])[0])
                answer = clean_string(answer)  # Clean the converted answer
            except Exception as e:
                print(f"Error processing answers[{i}]: {e}")
                print("Error Content:", answers[i])
                continue  # Skip to the next iteration if there's an error

            # Process tools based on the specified format
            if json_or_yaml == "json":
                functions = json.dumps(ast.literal_eval(tools[i][1:-1]), indent=1)
            elif json_or_yaml == "yaml":
                functions = json_to_yaml(tools[i])
            functions = clean_string(functions)  # Clean the functions string

            # Create and clean the message
            messages = _create_messages(queries[i], functions, answer)
            messages = sanitize_messages(messages)  # Clean all message contents

            # Append the processed prompt
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            print(f"Unicode error at example {i}: {e}")
            continue  # Skip examples that cause encoding errors

    return {"prompt": prompts}


def _convert_answer(answer):
    python_output = answer["name"] + "("
    for k, v in answer["arguments"].items():
        python_output += f"{k}={v},"
    python_output = python_output[:-1]
    python_output += ")"
    return python_output


def json_to_yaml(data):
    curr_func_yaml = ""
    json_func = ast.literal_eval(data)
    for func in json_func:
        curr_func_yaml += yaml.dump(func) + "\n\n"
    return curr_func_yaml


def _create_messages(user_prompt: str, functions: str, output: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in composing functions. You are given a question and a set of possible functions."
                " Based on the question, you will need to make one or more function/tool calls to achieve the purpose."
                " If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out."
                " You should only return the function call in tools call sections.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"#### Question: {user_prompt} "
                f"Here is a list of functions in JSON or YAML format that you can invoke:\n{functions}. "
                "Should you decide to return the function call(s), NO other text MUST be included.\n"
                "#### Response:"
            ),
        },
        {"role": "assistant", "content": output},
    ]
    return messages


def sanitize_messages(messages):
    """
    Sanitizes all 'content' fields in the messages to ensure they are UTF-8 compliant.
    """
    for message in messages:
        if "content" in message:
            message["content"] = clean_string(message["content"])
    return messages


def build_xlam_dataset(json_or_yaml: Literal["json", "yaml"], tokenizer):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/xlam-function-calling-60k")
    if json_or_yaml == "json":
        tokenized_ds = ds.map(
            functools.partial(process_xlam_data, tokenizer=tokenizer),
            batched=True,
            fn_kwargs={"json_or_yaml": "json"},
        )
    elif json_or_yaml == "yaml":
        tokenized_ds = ds.map(
            functools.partial(process_xlam_data, tokenizer=tokenizer),
            batched=True,
            fn_kwargs={"json_or_yaml": "yaml"},
        )
    tokenized_ds.save_to_disk(f"xlam_data_{json_or_yaml}")
    return tokenized_ds


if __name__ == "__main__":
    fire.Fire(run_finetune)
