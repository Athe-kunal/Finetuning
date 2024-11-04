import subprocess
import os
from finetune import run_evaluation, run_finetune
LLAMA_3_2_1B_INSTRUCT = "unsloth/Llama-3.2-1B-Instruct"
LLAMA_3_2_3B_INSTRUCT = "unsloth/Llama-3.2-3B-Instruct"
LLAMA_3_1_8B_INSTRUCT = "unsloth/llama-3-8b-Instruct"




if __name__ == "__main__":
    # subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_2_1B_INSTRUCT, "--dataset_name", "bfcl", "--json_or_yaml", "json"])
    # subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_2_1B_INSTRUCT, "--dataset_name", "bfcl", "--json_or_yaml", "yaml"])
    # subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_2_3B_INSTRUCT, "--dataset_name", "bfcl", "--json_or_yaml", "json"])
    # subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_2_3B_INSTRUCT, "--dataset_name", "bfcl", "--json_or_yaml", "yaml"])
    
    subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_1_8B_INSTRUCT, "--dataset_name", "bfcl", "--json_or_yaml", "yaml"])
    
    subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_2_1B_INSTRUCT, "--dataset_name", "xlam", "--json_or_yaml", "json"])
    subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_2_1B_INSTRUCT, "--dataset_name", "xlam", "--json_or_yaml", "yaml"])

    subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_2_3B_INSTRUCT, "--dataset_name", "xlam", "--json_or_yaml", "json"])
    subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_2_3B_INSTRUCT, "--dataset_name", "xlam", "--json_or_yaml", "yaml"])

    subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_1_8B_INSTRUCT, "--dataset_name", "xlam", "--json_or_yaml", "json"])
    subprocess.run(["poetry", "run", "python", "finetune.py", "--model_name", LLAMA_3_1_8B_INSTRUCT, "--dataset_name", "xlam", "--json_or_yaml", "yaml"])
    
    # subprocess.run(["poetry", "run", "python", "eval.py", "--model_name", LLAMA_3_2_1B_INSTRUCT,"--path", "outputs_unsloth/Llama-3.2-1B-Instruct_json_bfcl", "--dataset_name", "bfcl", "--json_or_yaml", "json"])
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-1B-Instruct_json_bfcl/epoch_1", model_name=LLAMA_3_2_1B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl")
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-1B-Instruct_json_bfcl/epoch_2", model_name=LLAMA_3_2_1B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl")
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-1B-Instruct_json_bfcl/epoch_3", model_name=LLAMA_3_2_1B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl")
    
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-3B-Instruct_json_bfcl/epoch_1", model_name=LLAMA_3_2_3B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl")
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-3B-Instruct_json_bfcl/epoch_2", model_name=LLAMA_3_2_3B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl")
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-3B-Instruct_json_bfcl/epoch_3", model_name=LLAMA_3_2_3B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl")

    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-3B-Instruct_yaml_bfcl/epoch_1", model_name=LLAMA_3_2_3B_INSTRUCT, json_or_yaml="yaml", dataset_name="bfcl")
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-3B-Instruct_yaml_bfcl/epoch_2", model_name=LLAMA_3_2_3B_INSTRUCT, json_or_yaml="yaml", dataset_name="bfcl")
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-3B-Instruct_yaml_bfcl/epoch_3", model_name=LLAMA_3_2_3B_INSTRUCT, json_or_yaml="yaml", dataset_name="bfcl")

    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-1B-Instruct_yaml_bfcl/epoch_1", model_name=LLAMA_3_2_1B_INSTRUCT, json_or_yaml="yaml", dataset_name="bfcl")
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-1B-Instruct_yaml_bfcl/epoch_2", model_name=LLAMA_3_2_1B_INSTRUCT, json_or_yaml="yaml", dataset_name="bfcl")
    # run_evaluation(model_path="outputs_unsloth/Llama-3.2-1B-Instruct_yaml_bfcl/epoch_3", model_name=LLAMA_3_2_1B_INSTRUCT, json_or_yaml="yaml", dataset_name="bfcl")

    # run_evaluation(model_path="outputs_unsloth/llama-3-8b-Instruct_json_bfcl/epoch_1", model_name=LLAMA_3_1_8B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl",val_batch_size=32)
    # run_evaluation(model_path="outputs_unsloth/llama-3-8b-Instruct_json_bfcl/epoch_2", model_name=LLAMA_3_1_8B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl",val_batch_size=32)
    # run_evaluation(model_path="outputs_unsloth/llama-3-8b-Instruct_json_bfcl/epoch_3", model_name=LLAMA_3_1_8B_INSTRUCT, json_or_yaml="json", dataset_name="bfcl",val_batch_size=32)

