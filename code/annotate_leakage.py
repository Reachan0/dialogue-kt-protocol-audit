"""Leakage-protocol (status quo) annotation: whole dialogue in ONE prompt, one JSON out.

This replicates the LLMKT (LAK 2025) full-dialogue correctness annotation
protocol, in which future turns are visible when labeling any given turn.
System prompt is the LLMKT ANNO_CORRECTNESS_SYSTEM_PROMPT verbatim, with the
dataset description adapted for QATD2k (multiple-choice anchored tutoring).

Red lines (per external review):
- one API call per dialogue, whole dialogue rendered at once;
- turn indices come from the SAME dialogues_{split}.jsonl as the causal
  pipeline, so labels join on (dialogue_id, turn);
- output files are separate from causal labels; neither enters the other's
  prompt context.

Usage:
  python annotate_leakage.py --split test --limit 6   # pilot
  python annotate_leakage.py --split test
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import anthropic

from annotate import MAIN_ANNOTATOR, turn_text

DATA = Path(__file__).resolve().parent.parent / "data"

QATD_DESC = ("the student is working on a multiple-choice math question. "
             "You are also given this question and its answer options.")

# LLMKT ANNO_CORRECTNESS_SYSTEM_PROMPT (dialogue-kt repo), verbatim apart from
# the {desc} substitution, exactly as in the original pipeline.
SYSTEM_PROMPT = f"""You are an experienced math teacher and education expert. You are given a dialogue between a student and teacher where {QATD_DESC} Your job is to determine if the student has correctly answered each of the teacher's questions in the dialogue. Please follow these instructions carefully when making your prediction:
- The prediction should be "na" if the teacher asks a question that 1) does not attempt to assess the student's knowledge, or 2) does not necessarily have a right or wrong answer.
- If the question 1) does assess student knowledge and has a correct answer, and 2) the student does not respond directly to the question, then the prediction should be "false".
- Before giving your final response, write a short summary of each turn, including the intended learning objectives, and explain why the student's response is right or wrong.
- Your final prediction should be a JSON object using the template: result = {{"turn 1": "true/false/na", "turn 2": ...}}"""

RESULT_RE = re.compile(r"result\s*=\s*(\{.*?\})", re.S)


def build_user_prompt(d: dict) -> tuple[str, list]:
    parts = []
    if d["question_context"]:
        parts.append(f"[BEGIN PROBLEM]\n{d['question_context']}\n[END PROBLEM]")
    lines, student_turns = [], []
    for idx, turn in enumerate(d["turns"], start=1):
        lines.append(turn_text(idx, turn))
        if turn["student"]:
            student_turns.append(idx)
    parts.append("[BEGIN DIALOGUE]\n" + "\n".join(lines) + "\n[END DIALOGUE]")
    turn_list = ", ".join(str(i) for i in student_turns)
    parts.append(f"Give one prediction for each of these student turns, and no others: {turn_list}.")
    return "\n\n".join(parts), student_turns


def parse_labels(text: str, student_turns: list) -> dict | None:
    candidates = []
    m = RESULT_RE.search(text)
    if m:
        candidates.append(m.group(1))
    # fallback: any {...} JSON objects, prefer the last one (final answer)
    candidates.extend(re.findall(r"\{[^{}]*\}", text)[::-1])
    obj = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(obj, dict):
        return None
    labels = {}
    for k, v in obj.items():
        km = re.search(r"(\d+)", str(k))
        if km and str(v).lower() in ("true", "false", "na"):
            labels[int(km.group(1))] = str(v).lower()
    if not all(t in labels for t in student_turns):
        return None
    return labels


async def annotate_dialogue(client, model, d, out_f, rej_f, lock, stats):
    user_prompt, student_turns = build_user_prompt(d)
    text, labels = None, None
    for attempt in range(3):
        resp = await client.messages.create(
            model=model,
            max_tokens=4096,
            thinking={"type": "disabled"},
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        labels = parse_labels(text, student_turns)
        u = resp.usage
        async with lock:
            stats["input"] = stats.get("input", 0) + (u.input_tokens or 0)
            stats["cache_read"] = stats.get("cache_read", 0) + (u.cache_read_input_tokens or 0)
            stats["output"] = stats.get("output", 0) + (u.output_tokens or 0)
            stats["calls"] = stats.get("calls", 0) + 1
        if labels is not None:
            break
    turns_by_idx = {i: t for i, t in enumerate(d["turns"], start=1)}
    async with lock:
        if labels is None:
            rej_f.write(json.dumps({"dialogue_id": d["dialogue_id"], "raw": text},
                                   ensure_ascii=False) + "\n")
            rej_f.flush()
            stats["parse_fail"] = stats.get("parse_fail", 0) + 1
            return
        for idx in student_turns:
            out_f.write(json.dumps({
                "dialogue_id": d["dialogue_id"], "split": d["split"], "turn": idx,
                "student": turns_by_idx[idx]["student"],
                "teacher": turns_by_idx[idx]["teacher"],
                "label": labels[idx],
            }, ensure_ascii=False) + "\n")
        out_f.write("")
        out_f.flush()
        stats["dialogues"] = stats.get("dialogues", 0) + 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=MAIN_ANNOTATOR)
    ap.add_argument("--out-suffix", default="", help="e.g. .fable5 for robustness relabeling")
    ap.add_argument("--ids-file", default=None, help="file with one dialogue_id per line; restrict to these")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    dialogues = [json.loads(l) for l in open(DATA / f"dialogues_{args.split}.jsonl", encoding="utf-8")]
    if args.ids_file:
        keep = set(l.strip() for l in open(args.ids_file) if l.strip())
        dialogues = [d for d in dialogues if d["dialogue_id"] in keep]
    out_path = DATA / f"leakage_labels_{args.split}{args.out_suffix}.jsonl"
    rej_path = DATA / f"leakage_rejects_{args.split}{args.out_suffix}.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            done.add(json.loads(l)["dialogue_id"])
    todo = [d for d in dialogues if d["dialogue_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.split}: {len(todo)} dialogues to annotate ({len(done)} done), model={args.model}")

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats: dict = {}
    t0 = time.time()

    async def run_one(d):
        async with sem:
            try:
                await annotate_dialogue(client, args.model, d, out_f, rej_f, lock, stats)
            except Exception as e:
                print(f"[ERROR] dialogue {d['dialogue_id']}: {type(e).__name__}: {str(e)[:120]}")

    with open(out_path, "a", encoding="utf-8") as out_f, open(rej_path, "a", encoding="utf-8") as rej_f:
        await asyncio.gather(*(run_one(d) for d in todo))

    print(f"\ndone in {time.time()-t0:.0f}s; stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
