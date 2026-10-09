"""QATD2k CSV -> dialogues.jsonl with (teacher, student) turn pairs + question context."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load_question_context() -> dict:
    by_question_intervention = defaultdict(list)
    with open(DATA / "dq_question_metadata.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["QuestionId_DQ"], row["InterventionId"])
            by_question_intervention[key].append(row)
    ctx = {}
    for key, rows in by_question_intervention.items():
        rows.sort(key=lambda r: int(r["Sequence"]))
        by_label = defaultdict(list)
        label_order = []
        for row in rows:
            label = row["Label"]
            text = row["Text"].strip()
            if not text:
                continue
            if label not in by_label:
                label_order.append(label)
            by_label[label].append(text)
        parts = [f"{label}: {' '.join(by_label[label])}" for label in label_order]
        ctx[key] = "\n".join(parts)
    return ctx


def messages_to_turns(msgs: list) -> list:
    """Merge consecutive same-role messages, pair into (teacher, student) turns."""
    blocks = []
    for m in msgs:
        role = "teacher" if m["IsTutor"] == "1" else "student"
        text = m["MessageString"].strip()
        if not text:
            continue
        if blocks and blocks[-1][0] == role:
            blocks[-1][1].append(text)
        else:
            blocks.append((role, [text]))
    turns, cur = [], {}
    for role, texts in blocks:
        text = "\n".join(texts)
        if role == "teacher":
            if cur:  # teacher again after a completed pair, or teacher-only turn pending
                turns.append(cur)
            cur = {"teacher": text, "student": None}
        else:
            if not cur:
                cur = {"teacher": None, "student": text}
            else:
                cur["student"] = text
            turns.append(cur)
            cur = {}
    if cur:
        turns.append(cur)
    return turns


def main():
    qctx = load_question_context()
    for split in ["train", "test"]:
        dlg = defaultdict(list)
        with open(DATA / f"qatd2k_{split}.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dlg[row["InterventionId"]].append(row)
        n_turns = 0
        out_path = DATA / f"dialogues_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as out:
            for did, msgs in dlg.items():
                msgs.sort(key=lambda m: int(m["MessageSequence"]))
                turns = messages_to_turns(msgs)
                student_turns = sum(1 for t in turns if t["student"])
                if student_turns == 0:
                    continue
                n_turns += student_turns
                qid = msgs[0]["QuestionId_DQ"]
                out.write(json.dumps({
                    "dialogue_id": did,
                    "split": split,
                    "question_id": qid,
                    "question_context": qctx.get((qid, did), ""),
                    "turns": turns,
                }, ensure_ascii=False) + "\n")
        print(f"{split}: wrote {sum(1 for _ in open(out_path))} dialogues, {n_turns} student turns -> {out_path}")


if __name__ == "__main__":
    main()
