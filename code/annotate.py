"""Causal turn-by-turn correctness annotation for QATD2k.

Protocol: the label for student turn j is produced from history <= j only
(strictly causal / online observation protocol). Within a dialogue, calls are
prefix-incremental: each turn appends to a growing conversation, so the prompt
prefix is byte-stable and prompt-cacheable. The shared system prompt carries a
cache_control breakpoint and is reused across all ~28k calls.

Usage:
  python annotate.py --split test --limit 4 --out-suffix .pilot
  python pilot_history_ablation.py                    # paired pilot, dry run
  python pilot_history_ablation.py --execute          # isolated original / labels-only runs
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import anthropic

DATA = Path(__file__).resolve().parent.parent / "data"

# ===== 模型锁定 (grok review 工程项 2) =====
# 主标注(因果 + 泄漏两种协议): gpt-5.6-sol
# 鲁棒性重标(~50 段/500 轮,双协议): claude-fable-5 (--model claude-fable-5 --out-suffix .fable5)
MAIN_ANNOTATOR = "gpt-5.6-sol"
ROBUSTNESS_ANNOTATOR = "claude-fable-5"

# Adapted from LLMKT's ANNO_CORRECTNESS_SYSTEM_PROMPT (dialogue-kt repo), changed
# from whole-dialogue annotation to strictly-causal per-turn annotation. Two
# worked examples are included both for label quality and to grow the shared
# cacheable prefix.
SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are shown a live tutoring dialogue between a student and a teacher, one turn at a time, as it unfolds. The student is working on a multiple-choice math question, which you are also given. Your job is to judge, after each student turn, whether the student correctly answered the teacher's most recent question.

You only ever see the dialogue up to the current turn. You must never assume anything about how the dialogue will continue.

Follow these rules for every judgment:
- The prediction should be "na" if the teacher's most recent utterance 1) does not attempt to assess the student's knowledge (e.g., greetings, encouragement, scheduling), or 2) does not have a right or wrong answer (e.g., "would you like a hint?").
- If the teacher's question does assess knowledge and has a correct answer, but the student does not respond directly to it, the prediction is "false".
- Judge only the student's most recent turn, using the question, the answer options, and the dialogue so far as context.
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

RESULT_RE = re.compile(r'result\s*=\s*\{\s*"correct"\s*:\s*"(true|false|na)"\s*\}')


def turn_text(idx: int, turn: dict) -> str:
    lines = []
    if turn["teacher"]:
        lines.append(f"Teacher Turn {idx}: {turn['teacher']}")
    if turn["student"]:
        lines.append(f"Student Turn {idx}: {turn['student']}")
    return "\n".join(lines)


def assistant_history(raw: str, label: str, history_mode: str) -> str:
    """Keep current-turn generation unchanged; control only later replay."""
    if history_mode == "original":
        return raw
    if history_mode == "labels-only" and label in {"true", "false", "na"}:
        return 'result = ' + json.dumps({"correct": label})
    raise ValueError("Invalid history mode or missing label for labels-only history")


async def annotate_dialogue(client, model: str, d: dict, out_f, rej_f, lock, stats, done_turns,
                            history_mode="original", stop_on_parse_failure=False):
    """done_turns: {turn_idx: record} — 已标注轮,重放而不调 API;遇到缺口后转为在线标注。"""
    if history_mode not in {"original", "labels-only"}:
        raise ValueError(f"Unknown history mode: {history_mode}")
    messages = []  # append-only conversation; cache_control added per-request
    pending = []   # context accumulated since the last judged student turn
    replaying = True
    if d["question_context"]:
        pending.append(f"[BEGIN PROBLEM]\n{d['question_context']}\n[END PROBLEM]")
    pending.append("[DIALOGUE START]")
    for idx, turn in enumerate(d["turns"], start=1):
        pending.append(turn_text(idx, turn))
        if not turn["student"]:
            continue  # teacher-only turn: fold into the next user message
        user_text = "\n\n".join(pending) + f"\n\nJudge student turn {idx}."
        pending = []
        messages.append({"role": "user", "content": user_text})

        if replaying and idx in done_turns:
            # 从已写 raw 重建 assistant 历史 (grok review 工程项 1)
            rec = done_turns[idx]
            messages.append({"role": "assistant", "content": assistant_history(
                rec["raw"], rec["label"], history_mode)})
            continue
        replaying = False

        # exactly 2 cache breakpoints per request: system + last user block
        req_msgs = messages[:-1] + [{
            "role": "user",
            "content": [{"type": "text", "text": user_text,
                         "cache_control": {"type": "ephemeral"}}],
        }]
        text, label = None, None
        for attempt in range(3):  # parse-fail 重试 (grok review 工程项 4)
            resp = await client.messages.create(
                model=model,
                max_tokens=1024,
                thinking={"type": "disabled"},
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=req_msgs,
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            m = RESULT_RE.search(text)
            if m:
                label = m.group(1)
            u = resp.usage
            usage = {
                "input": u.input_tokens or 0,
                "cache_read": u.cache_read_input_tokens or 0,
                "cache_creation": u.cache_creation_input_tokens or 0,
                "output": u.output_tokens or 0,
            }
            async with lock:
                for k, v in usage.items():
                    stats[k] = stats.get(k, 0) + v
                stats["calls"] = stats.get("calls", 0) + 1
            if label is not None:
                break
        rec = {
            "dialogue_id": d["dialogue_id"], "split": d["split"], "turn": idx,
            "student": turn["student"], "teacher": turn["teacher"],
            "label": label, "raw": text, "usage": usage,
        }
        async with lock:
            if label is None:
                # 不进评测文件 (grok review 工程项 4)
                rej_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rej_f.flush()
                stats["parse_fail"] = stats.get("parse_fail", 0) + 1
            else:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
        if label is None and (stop_on_parse_failure or history_mode == "labels-only"):
            raise ValueError(f"Unparsed label at dialogue {d['dialogue_id']} turn {idx}")
        # Store full raw output above, but control what later turns can see.
        messages.append({"role": "assistant", "content": assistant_history(text, label, history_mode)})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None, help="max dialogues (pilot)")
    ap.add_argument("--model", default=MAIN_ANNOTATOR)
    ap.add_argument("--out-suffix", default="", help="e.g. .fable5 for robustness relabeling")
    ap.add_argument("--ids-file", default=None, help="file with one dialogue_id per line; restrict to these")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    dialogues = [json.loads(l) for l in open(DATA / f"dialogues_{args.split}.jsonl", encoding="utf-8")]
    if args.ids_file:
        keep = set(l.strip() for l in open(args.ids_file) if l.strip())
        dialogues = [d for d in dialogues if d["dialogue_id"] in keep]
    out_path = DATA / f"causal_labels_{args.split}{args.out_suffix}.jsonl"
    rej_path = DATA / f"causal_rejects_{args.split}{args.out_suffix}.jsonl"
    # 按 (dialogue_id, turn) 粒度续跑 (grok review 工程项 1)
    done: dict = {}
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            done.setdefault(r["dialogue_id"], {})[r["turn"]] = r
    total_st = {d["dialogue_id"]: sum(1 for t in d["turns"] if t["student"]) for d in dialogues}
    todo = [d for d in dialogues
            if len(done.get(d["dialogue_id"], {})) < total_st[d["dialogue_id"]]]
    if args.limit:
        todo = todo[: args.limit]
    n_done_turns = sum(len(v) for v in done.values())
    print(f"{args.split}: {len(todo)} dialogues to (re)process, {n_done_turns} turns already labeled, model={args.model}")

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats: dict = {}
    t0 = time.time()

    async def run_one(d):
        async with sem:
            try:
                await annotate_dialogue(client, args.model, d, out_f, rej_f, lock, stats,
                                        done.get(d["dialogue_id"], {}))
            except Exception as e:
                print(f"[ERROR] dialogue {d['dialogue_id']}: {type(e).__name__}: {str(e)[:120]}")

    with open(out_path, "a", encoding="utf-8") as out_f, open(rej_path, "a", encoding="utf-8") as rej_f:
        await asyncio.gather(*(run_one(d) for d in todo))

    el = time.time() - t0
    print(f"\ndone in {el:.0f}s; stats: {stats}")
    if stats.get("calls"):
        print("token totals — 计费以代理为准 (gpt-5.6-sol/claude-opus-5 单价未知,不做美元估算)")
        print(f"cache hit volume: read={stats.get('cache_read', 0)} vs uncached input={stats.get('input', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
