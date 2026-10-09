"""Prefix vs Full correctness annotation on MathDial (second-dataset replication).

Both protocols run on the dialogue-kt MathDial CSVs (data/annotated/mathdial_{split}_atc.csv),
so turn indices are exactly the ones the KT pipeline uses.

- prefix: turn j is judged from the problem context + turns <= j only, one call per
  student turn, prompt grows incrementally (same design as annotate.py on QATD2k).
- full:   whole dialogue in one call, LLMKT ANNO_CORRECTNESS_SYSTEM_PROMPT verbatim with the
  LLMKT MathDial description (the original LLMKT pipeline on its original dataset).

Credentials come only from GATEWAY_BASE_URL / GATEWAY_API_KEY (see run_mathdial.sh);
inherited ANTHROPIC_* variables are removed before the client is built so no other token
can be sent to the gateway.

Usage:
  python annotate_mathdial.py --protocol prefix --split test --limit 5 --run-dir RUN
  python annotate_mathdial.py --protocol full --split test --run-dir RUN --max-calls 700
"""
import argparse
import ast
import asyncio
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from annotate import RESULT_RE as PREFIX_RESULT_RE, turn_text
from annotate_leakage import parse_labels

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dialogue-kt" / "data" / "annotated"
csv.field_size_limit(sys.maxsize)

MATHDIAL_DESC = ("the student is attempting to solve a math problem. You are also given this problem, "
                 "its correct solution, and the incorrect solution the student initially gave.")

# LLMKT ANNO_CORRECTNESS_SYSTEM_PROMPT (dialogue_kt/prompting.py), verbatim with the MathDial desc.
FULL_SYSTEM_PROMPT = f"""You are an experienced math teacher and education expert. You are given a dialogue between a student and teacher where {MATHDIAL_DESC} Your job is to determine if the student has correctly answered each of the teacher's questions in the dialogue. Please follow these instructions carefully when making your prediction:
- The prediction should be "na" if the teacher asks a question that 1) does not attempt to assess the student's knowledge, or 2) does not necessarily have a right or wrong answer.
- If the question 1) does assess student knowledge and has a correct answer, and 2) the student does not respond directly to the question, then the prediction should be "false".
- Do not give a prediction for turn 0 since the teacher does not ask a question.
- Before giving your final response, write a short summary of each turn, including the intended learning objectives, and explain why the student's response is right or wrong.
- Your final prediction should be a JSON object using the template: result = {{"turn 1": "true/false/na", "turn 2": ...}}"""

# annotate.SYSTEM_PROMPT with the QATD2k task description replaced by the MathDial one; rules
# and worked examples are unchanged so the correctness rubric matches the QATD2k prefix protocol.
PREFIX_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are shown a live tutoring dialogue between a student and a teacher, one turn at a time, as it unfolds. The student is attempting to solve a math problem; you are also given this problem, its correct solution, and the incorrect solution the student initially gave. Your job is to judge, after each student turn, whether the student correctly answered the teacher's most recent question.

You only ever see the dialogue up to the current turn. You must never assume anything about how the dialogue will continue.

Follow these rules for every judgment:
- The prediction should be "na" if the teacher's most recent utterance 1) does not attempt to assess the student's knowledge (e.g., greetings, encouragement, scheduling), or 2) does not have a right or wrong answer (e.g., "would you like a hint?").
- If the teacher's question does assess knowledge and has a correct answer, but the student does not respond directly to it, the prediction is "false".
- Judge only the student's most recent turn, using the problem, the solutions, and the dialogue so far as context.
- Before your final answer, write a one-sentence summary of what the teacher asked and what the student's reply means, then explain briefly why the reply is right or wrong.
- End your response with exactly one line of the form: result = {"correct": "true"} (or "false" or "na").

Example 1:
[BEGIN PROBLEM]
Question Text: What is \\( \\frac{1}{2} + \\frac{1}{4} \\)?
Answer A Text: \\( \\frac{2}{6} \\)
Answer B Text: \\( \\frac{3}{4} \\)
[END PROBLEM]
Teacher: To add these fractions, what do we need the denominators to be?
Student: they need to be the same, so 4
Reasoning: The teacher asked what the denominators must be for addition; the student said they must be equal and identified 4 as the common denominator, which is correct.
result = {"correct": "true"}

Example 2:
[BEGIN PROBLEM]
Question Text: Round 5.4598 to 1 decimal place.
Answer A Text: 5.5
Answer B Text: 5.4
[END PROBLEM]
Teacher: Which digit do we look at to decide whether to round up or down?
Student: idk can you just tell me the answer
Reasoning: The teacher asked a knowledge-assessing question about the deciding digit; the student did not attempt to answer it.
result = {"correct": "false"}"""


def mathdial_context(meta: dict) -> str:
    # Same rendering as dialogue_kt.prompting.get_mathdial_context.
    return (f"[BEGIN PROBLEM]\n{meta['question'].strip()}\n[END PROBLEM]\n\n"
            f"[BEGIN CORRECT SOLUTION]\n{meta['correct_solution'].strip()}\n[END CORRECT SOLUTION]\n\n"
            f"[BEGIN INCORRECT STUDENT SOLUTION]\n{meta['incorrect_solution'].strip()}\n[END INCORRECT STUDENT SOLUTION]")


def load_dialogues(split: str) -> list:
    out = []
    with open(SRC / f"mathdial_{split}_atc.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({
                "dialogue_id": row["index"],
                "split": split,
                "meta": ast.literal_eval(row["meta_data"]),
                "turns": ast.literal_eval(row["dialogue"]),
            })
    return out


def make_client():
    for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        os.environ.pop(var, None)
    base_url, key = os.environ.get("GATEWAY_BASE_URL"), os.environ.get("GATEWAY_API_KEY")
    if not base_url or not key:
        raise SystemExit("GATEWAY_BASE_URL and GATEWAY_API_KEY must be set")
    return anthropic.AsyncAnthropic(api_key=key, base_url=base_url, max_retries=0, timeout=300)


class Budget:
    """Hard cap on API attempts across the run, including retries and resumes."""

    def __init__(self, run_dir: Path, max_calls: int):
        self.path = run_dir / "attempts.jsonl"
        self.max_calls = max_calls
        self.used = sum(1 for _ in open(self.path)) if self.path.exists() else 0

    def take(self, meta: dict):
        if self.used >= self.max_calls:
            raise RuntimeError(f"call budget {self.max_calls} exhausted")
        self.used += 1
        with open(self.path, "a") as f:
            f.write(json.dumps({"n": self.used, "t": datetime.now(timezone.utc).isoformat(), **meta}) + "\n")


async def call(client, budget, lock, stats, model, system, messages, max_tokens, meta):
    """One request with at most one retry for transient errors; returns (text, usage) or raises."""
    for attempt in range(2):
        async with lock:
            budget.take({**meta, "attempt": attempt})
        try:
            resp = await client.messages.create(
                model=model, max_tokens=max_tokens, thinking={"type": "disabled"},
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
            )
        except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError,
                anthropic.InternalServerError) as e:
            if attempt == 1:
                raise
            await asyncio.sleep(5 if isinstance(e, anthropic.RateLimitError) else 1)
            continue
        if resp.model != model:
            raise RuntimeError(f"returned model {resp.model!r} != requested {model!r}")
        u = resp.usage
        usage = {"input": u.input_tokens or 0, "output": u.output_tokens or 0,
                 "cache_read": u.cache_read_input_tokens or 0,
                 "cache_creation": u.cache_creation_input_tokens or 0}
        async with lock:
            for k, v in usage.items():
                stats[k] = stats.get(k, 0) + v
            stats["calls"] = stats.get("calls", 0) + 1
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, usage, resp.id, resp.stop_reason
    raise AssertionError("unreachable")


async def prefix_dialogue(client, budget, lock, stats, model, d, done, out_f, rej_f):
    messages, first = [], True
    for turn in d["turns"]:
        idx = turn["turn"]
        text = turn_text(idx, turn)
        if first:
            text = mathdial_context(d["meta"]) + "\n\n[DIALOGUE START]\n" + text
            first = False
        user_text = text + f"\n\nJudge student turn {idx}."
        messages.append({"role": "user", "content": user_text})
        if idx in done:
            messages.append({"role": "assistant", "content": done[idx]["raw"]})
            continue
        req = messages[:-1] + [{"role": "user", "content": [
            {"type": "text", "text": user_text, "cache_control": {"type": "ephemeral"}}]}]
        label, raw = None, None
        for parse_try in range(2):
            raw, usage, rid, stop = await call(client, budget, lock, stats, model, PREFIX_SYSTEM_PROMPT,
                                               req, 1024, {"dialogue_id": d["dialogue_id"], "turn": idx,
                                                           "parse_try": parse_try})
            m = PREFIX_RESULT_RE.search(raw)
            if m:
                label = m.group(1)
                break
        rec = {"dialogue_id": d["dialogue_id"], "split": d["split"], "turn": idx, "label": label,
               "raw": raw, "usage": usage, "response_id": rid, "stop_reason": stop}
        async with lock:
            (out_f if label else rej_f).write(json.dumps(rec, ensure_ascii=False) + "\n")
            (out_f if label else rej_f).flush()
        if label is None:
            # Later turns would condition on an unparsed judgment; stop this dialogue.
            raise ValueError(f"unparsed label at turn {idx}")
        messages.append({"role": "assistant", "content": raw})


async def full_dialogue(client, budget, lock, stats, model, d, out_f, rej_f):
    student_turns = [t["turn"] for t in d["turns"] if t["student"]]
    lines = "\n".join(turn_text(t["turn"], t) for t in d["turns"])
    prompt = (mathdial_context(d["meta"]) + "\n\n[BEGIN DIALOGUE]\n" + lines + "\n[END DIALOGUE]\n\n"
              + f"Give one prediction for each of these student turns, and no others: "
              + ", ".join(map(str, student_turns)) + ".")
    labels, raw = None, None
    for parse_try in range(2):
        raw, usage, rid, stop = await call(client, budget, lock, stats, model, FULL_SYSTEM_PROMPT,
                                           [{"role": "user", "content": prompt}], 4096,
                                           {"dialogue_id": d["dialogue_id"], "parse_try": parse_try})
        labels = parse_labels(raw, student_turns)
        if labels is not None:
            break
    async with lock:
        if labels is None:
            rej_f.write(json.dumps({"dialogue_id": d["dialogue_id"], "raw": raw}, ensure_ascii=False) + "\n")
            rej_f.flush()
            return
        for idx in student_turns:
            out_f.write(json.dumps({"dialogue_id": d["dialogue_id"], "split": d["split"], "turn": idx,
                                    "label": labels[idx], "response_id": rid, "stop_reason": stop},
                                   ensure_ascii=False) + "\n")
        out_f.write(json.dumps({"dialogue_id": d["dialogue_id"], "split": d["split"], "turn": None,
                                "raw": raw, "usage": usage}, ensure_ascii=False) + "\n")
        out_f.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["prefix", "full"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    run_dir = args.run_dir / f"{args.protocol}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path, rej_path = run_dir / "labels.jsonl", run_dir / "rejects.jsonl"
    dialogues = load_dialogues(args.split)
    if args.limit:
        dialogues = dialogues[: args.limit]

    done: dict = {}
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            if r["turn"] is not None:
                done.setdefault(r["dialogue_id"], {})[r["turn"]] = r
    n_turns = {d["dialogue_id"]: sum(1 for t in d["turns"] if t["student"]) for d in dialogues}
    todo = [d for d in dialogues if len(done.get(d["dialogue_id"], {})) < n_turns[d["dialogue_id"]]]

    budget = Budget(run_dir, args.max_calls)
    print(f"{args.protocol}/{args.split}: {len(todo)}/{len(dialogues)} dialogues to do, "
          f"budget used {budget.used}/{args.max_calls}, model={args.model}", flush=True)
    client = make_client()
    sem, lock, stats, errors = asyncio.Semaphore(args.concurrency), asyncio.Lock(), {}, []
    t0 = time.time()

    async def run_one(d):
        async with sem:
            try:
                if args.protocol == "prefix":
                    await prefix_dialogue(client, budget, lock, stats, args.model, d,
                                          done.get(d["dialogue_id"], {}), out_f, rej_f)
                else:
                    await full_dialogue(client, budget, lock, stats, args.model, d, out_f, rej_f)
            except Exception as e:
                errors.append(d["dialogue_id"])
                print(f"[ERROR] {d['dialogue_id']}: {type(e).__name__}: {str(e)[:160]}", flush=True)

    with open(out_path, "a", encoding="utf-8") as out_f, open(rej_path, "a", encoding="utf-8") as rej_f:
        await asyncio.gather(*(run_one(d) for d in todo))

    summary = {"protocol": args.protocol, "split": args.split, "model": args.model,
               "dialogues": len(dialogues), "errors": len(errors), "seconds": round(time.time() - t0),
               "budget_used": budget.used, **stats}
    with open(run_dir / "invocations.jsonl", "a") as f:
        f.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(), **summary}) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
