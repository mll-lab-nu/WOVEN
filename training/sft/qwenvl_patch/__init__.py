import os
import re

# Dataset registry for qwen-vl-finetune.
#
# `--dataset_use` / `--eval_dataset_use` accept comma-separated entries, each either
#   * a path to a .json / .jsonl annotation file (e.g. the output of training/prepare_data.py), or
#   * a registered name below: "woven_train" reads $WOVEN_SFT_TRAIN_JSON, "woven_val" reads
#     $WOVEN_SFT_VAL_JSON.
# An optional "%NN" suffix samples NN percent of the entries.
# Relative image paths inside an annotation file are resolved against the file's directory.

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
}

# Registered names that resolve to an annotation file given by an environment variable.
ENV_DATASETS = {
    "woven_train": "WOVEN_SFT_TRAIN_JSON",
    "woven_val": "WOVEN_SFT_VAL_JSON",
}


def _file_config(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise ValueError(f"annotation file not found: {path}")
    return {"annotation_path": path, "data_path": os.path.dirname(path)}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in ENV_DATASETS:
            env = ENV_DATASETS[dataset_name]
            if not os.environ.get(env):
                raise ValueError(f"dataset {dataset_name} requires ${env}")
            config = _file_config(os.environ[env])
        elif dataset_name.endswith((".json", ".jsonl")):
            config = _file_config(dataset_name)
        elif dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
        else:
            raise ValueError(f"do not find {dataset_name}")
        config["sampling_rate"] = sampling_rate
        config_list.append(config)
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
