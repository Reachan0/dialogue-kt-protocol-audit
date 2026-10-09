from typing import List, Union
from copy import deepcopy
import json
import re
from ast import literal_eval
import pandas as pd

COMTA_SUBJECTS = ["Elementary", "Algebra", "Trigonometry", "Geometry"]

def add_content(cur: str, new: str):
    new = new.strip()
    if not cur:
        return new
    if cur == new: # Sometimes turns are repeated in MathDial
        return cur
    if not cur.endswith((".", "!", "?")):
        cur += "."
    return cur + " " + new

def process_dialogue(turns: List[dict]):
    cur_role = turns[0]["role"]
    cur_turn = {
        "turn": 0 if cur_role == "student" else 1,
        "teacher": "",
        "student": ""
    }
    result = []
    for turn in turns:
        if turn["role"] == "teacher" and cur_role == "student":
            result.append(cur_turn)
            cur_turn = {
                "turn": cur_turn["turn"] + 1,
                "teacher": "",
                "student": ""
            }
        cur_role = turn["role"]
        cur_turn[cur_role] = add_content(cur_turn[cur_role], turn["content"])
    # Only include final turn if there was a student response (not always the case in MathDial)
    if cur_turn["student"]:
        result.append(cur_turn)
    return result

def correct_from_str(correct: str):
    return True if correct == "true" else False if correct == "false" else None

def correct_to_str(correct: Union[bool, None]):
    return "na" if correct is None else "true" if correct else False

def standards_to_str(standards: List[str], sep: str):
    return "None" if not standards else sep.join([f"{idx + 1}) {kc}" for idx, kc in enumerate(standards)])

def load_comta_src_data():
    with open("data/src/CoMTA_dataset.json") as file:
        data = json.load(file)
    proc_data = []
    for index, sample in enumerate(data):
        # Skip calculus since not in ATC
        if sample["math_level"] == "Calculus":
            continue
        # Add dialogue and meta data
        proc_data.append({
            "index": index,
            "dialogue": process_dialogue([
                {"role": "student" if turn["role"] == "user" else "teacher", "content": turn["content"]}
                for turn in sample["data"]
            ]),
            "meta_data": {
                "expected_result": sample["expected_result"],
                "math_level": sample["math_level"]
            }
        })
    return pd.DataFrame(proc_data)

def load_mathdial_src_data(split: str):
    turn_prefix_re = re.compile(r"^[a-zA-Z]+: (\([a-z]+\))?")
    with open(f"data/src/mathdial/data/{split}.jsonl") as file:
        data = [json.loads(line) for line in file]
    proc_data = []
    for index, sample in enumerate(data):
        if not sample["self-typical-confusion"] or not sample["self-typical-interactions"]:
            continue
        # Add dialogue and meta data
        proc_data.append({
            "index": index,
            **sample,
            "dialogue": process_dialogue([
                {"role": "teacher" if turn.startswith("Teacher") else "student", "content": turn_prefix_re.sub("", turn)}
                for turn in sample["conversation"].split("|EOM|")
            ]),
            "meta_data": {
                "question": sample["question"],
                "correct_solution": sample["ground_truth"],
                "incorrect_solution": sample["student_incorrect_solution"],
                "self_correctness": sample["self-correctness"],
                "self_typical_confusion": sample["self-typical-confusion"],
                "self_typical_interactions": sample["self-typical-interactions"]
            }
        })
    return pd.DataFrame(proc_data)

def load_src_data(args, split: str = ""):
    if args.dataset == "comta":
        return load_comta_src_data()
    elif args.dataset == "mathdial":
        return load_mathdial_src_data(split)
    raise Exception(f"Loading not supported for {args.dataset}")

def get_annotated_data_filename(args, split: str = "", tag_src: str = None):
    protocol = tag_src or args.tag_src
    return f"data/annotated/{args.dataset}{f'_{split}' if split else ''}_{protocol}.csv"

def get_kc_dict_filename(args):
    if args.dataset == "qatd2k":
        return "data/annotated/kc_dict_qatd2k_eedi.json"
    if args.dataset == "mathdial" and args.tag_src in ("causal", "leakage"):
        # Protocol runs only relabel correctness; KCs stay LLMKT's ATC standards.
        return "data/annotated/kc_dict_mathdial_atc.json"
    return f"data/annotated/kc_dict_{args.dataset}_{args.tag_src}.json"

def load_kc_dict(args):
    with open(get_kc_dict_filename(args)) as file:
        return json.load(file)

def get_default_fold(args):
    if args.dataset == "comta":
        if args.split_by_subject:
            return "Elementary"
        return 1
    else:
        return None

def mask_to_common_non_na(df: pd.DataFrame, other_df: pd.DataFrame):
    other_by_id = dict(zip(other_df["dialogue_id"].astype(str), other_df["annotation"]))
    result = df.copy()
    for idx, row in result.iterrows():
        annotation = deepcopy(row["annotation"])
        other = other_by_id[str(row["dialogue_id"])]
        assert annotation.keys() == other.keys()
        for turn_key, turn in annotation.items():
            own_valid = turn["correct"] is not None and bool(turn["kcs"])
            other_valid = other[turn_key]["correct"] is not None and bool(other[turn_key]["kcs"])
            if not (own_valid and other_valid):
                turn["correct"] = None
                turn["kcs"] = []
        result.at[idx, "annotation"] = annotation
    return result


def load_annotated_data(args, fold: Union[int, str, None] = 1):
    if args.dataset == "comta":
        df = pd.read_csv(get_annotated_data_filename(args), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        if args.split_by_subject:
            assert fold in COMTA_SUBJECTS
            subj_mask = df.apply(lambda row: row["meta_data"]["math_level"] == fold, axis=1)
            test_df = df[subj_mask]
            train_df = df[~subj_mask].sample(frac=1, random_state=221)
            return (
                train_df[:int(.8 * len(train_df))],
                train_df[int(.8 * len(train_df)):],
                test_df
            )
        else:
            assert fold in range(1, 6)
            df = df.sample(frac=1, random_state=221)
            split_point = int(len(df) * ((fold - 1) / 5))
            df = pd.concat([df[split_point:], df[:split_point]])
            return (
                df[:int(len(df) * .65)],
                df[int(len(df) * .65) : int(len(df) * .8)],
                df[int(len(df) * .8):],
            )
    elif args.dataset == "mathdial":
        def pass_typical_threshold(row):
            return (row["meta_data"]["self_typical_confusion"] >= args.typical_cutoff and
                    row["meta_data"]["self_typical_interactions"] >= args.typical_cutoff)

        # Protocol CSVs share rows and order, so the seeded shuffle gives the same validation split.
        eval_tag_src = getattr(args, "eval_tag_src", None) or args.tag_src
        train_df = pd.read_csv(get_annotated_data_filename(args, "train"), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        train_df = train_df.sample(frac=1, random_state=221)
        train_df = train_df[train_df.apply(pass_typical_threshold, axis=1)]
        test_df = pd.read_csv(get_annotated_data_filename(args, "test", eval_tag_src), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        test_df = test_df[test_df.apply(pass_typical_threshold, axis=1)]
        return (
            train_df[:int(.8 * len(train_df))],
            train_df[int(.8 * len(train_df)):],
            test_df
        )
    elif args.dataset == "qatd2k":
        assert args.tag_src in ("causal", "leakage")
        eval_tag_src = getattr(args, "eval_tag_src", None) or args.tag_src
        assert eval_tag_src in ("causal", "leakage")
        converters = {col: literal_eval for col in ["dialogue", "meta_data", "annotation"]}
        train_df = pd.read_csv(get_annotated_data_filename(args, "train", args.tag_src), converters=converters)
        test_df = pd.read_csv(get_annotated_data_filename(args, "test", eval_tag_src), converters=converters)
        with open("data/annotated/qatd2k_val_dialogue_ids.json") as file:
            val_ids = set(json.load(file))
        val_mask = train_df["dialogue_id"].astype(str).isin(val_ids)
        # Deterministic order, shared across causal/leakage protocols.
        val_df = train_df[val_mask].sort_values("dialogue_id", key=lambda s: s.astype(str))
        train_df = train_df[~val_mask].sort_values("dialogue_id", key=lambda s: s.astype(str))
        test_df = test_df.sort_values("dialogue_id", key=lambda s: s.astype(str))

        if getattr(args, "train_common_mask", False):
            other_tag_src = "leakage" if args.tag_src == "causal" else "causal"
            other_train_df = pd.read_csv(get_annotated_data_filename(args, "train", other_tag_src), converters=converters)
            other_val_mask = other_train_df["dialogue_id"].astype(str).isin(val_ids)
            other_val_df = other_train_df[other_val_mask]
            other_train_df = other_train_df[~other_val_mask]
            train_df = mask_to_common_non_na(train_df, other_train_df)
            val_df = mask_to_common_non_na(val_df, other_val_df)

        history_tag_src = getattr(args, "history_tag_src", None)
        if getattr(args, "prompt_inc_labels", False) and history_tag_src:
            history_train = pd.read_csv(get_annotated_data_filename(args, "train", history_tag_src), converters=converters)
            history_test = pd.read_csv(get_annotated_data_filename(args, "test", history_tag_src), converters=converters)
            train_history = dict(zip(history_train["dialogue_id"].astype(str), history_train["annotation"]))
            test_history = dict(zip(history_test["dialogue_id"].astype(str), history_test["annotation"]))
            for df, lookup in [(train_df, train_history), (val_df, train_history), (test_df, test_history)]:
                df["history_annotation"] = df["dialogue_id"].astype(str).map(lookup)
                assert df["history_annotation"].notna().all()
        return train_df, val_df, test_df
    raise Exception(f"Loading not supported for {args.dataset}")

def get_model_file_suffix(args, fold = None):
    suffix = "_incfirst" if args.inc_first_label else ""
    suffix += f"_{fold}" if fold else ""
    if args.model_name:
        return args.model_name + suffix
    if args.model_type == "lmkt": # Zero-shot LLMKT
        model_name = args.base_model.replace("/", "-")
    else:
        model_name = args.model_type
    return f"{args.dataset}_{model_name}_agg{args.agg}" + suffix

def get_kc_result_filename(args, fold):
    return f"results/kcs_{get_model_file_suffix(args, fold)}.json"

def get_prediction_result_filename(args, fold=None):
    eval_protocol = getattr(args, "eval_tag_src", None) or getattr(args, "tag_src", "")
    return f"results/predictions_{get_model_file_suffix(args, fold)}_eval-{eval_protocol}.csv"


def get_qual_result_filename(args, fold = None):
    return f"results/qual_{get_model_file_suffix(args, fold)}.csv"

def load_atc():
    with open("data/src/ATC/domain_groups.json") as file:
        domain_groups = json.load(file)

    with open("data/src/ATC/standards.jsonl") as file:
        standards = [json.loads(line) for line in file]

    for stand in standards:
        stand["description"] = stand["description"].split("\nGrade")[0] # Remove grade-level descriptions
        stand["description"] = stand["description"].replace("\n", " ") # Remove newlines for easier LM prompting

    return {
        "domain_groups": domain_groups,
        "standards": {tag["id"]: tag for tag in standards}
    }
