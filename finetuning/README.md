# FINETUNING JSON vs. YAML

To get started, first install the dependencies:

```bash
poetry install
```

Then  change your directory to the finetuning folder:

```bash
cd finetuning
```

You can save the XLAM dataset to disk by running:

```python
from finetune import build_xlam_dataset
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct", 
        dtype=torch.bfloat16,
        load_in_4bit=False,
        trust_remote_code=True,
)
build_xlam_dataset(json_or_yaml='json',tokenizer=tokenizer)
build_xlam_dataset(json_or_yaml='yaml',tokenizer=tokenizer)
```

Also, set environment variables for your wandb account:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export WANDB_PROJECT=<your_wandb_project>
export WANDB_ENTITY=<your_wandb_entity>
```

Then you can finetune the model by running:

```python
from finetune import run_finetune

run_finetune(model_name="unsloth/Llama-3.2-1B-Instruct", dataset_name="xlam", json_or_yaml="json")
run_finetune(model_name="unsloth/Llama-3.2-1B-Instruct", dataset_name="xlam", json_or_yaml="yaml")
```

Now, to run evaluation, you can run:

```python
from finetune import run_evaluation
run_evaluation(model_path="outputs_unsloth/Llama-3.2-1B-Instruct_json_xlam/epoch_1", model_name="unsloth/Llama-3.2-1B-Instruct", json_or_yaml="json", dataset_name="xlam")
```
