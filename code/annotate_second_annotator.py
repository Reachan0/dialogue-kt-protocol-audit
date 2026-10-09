"""E2: re-run both QATD2k protocols on the test split with a second annotator model.

Prompts, request construction and parsing are the unchanged annotate.annotate_dialogue
(Prefix) and annotate_leakage.annotate_dialogue (Full); only the client differs. The
client wrapper enforces a hard attempt budget, retries a transient error once, checks
the returned model name and logs every response in a ledger. Thinking blocks, if the
model returns them despite thinking being disabled, are counted but never replayed:
both annotators build history from text blocks only.

Usage:
  ./run_e2.sh --protocol prefix --run-dir RUN --max-calls 6000 [--limit 3]
"""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

import annotate
import annotate_leakage
from annotate_mathdial import Budget, make_client

TRANSIENT = (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError,
             anthropic.InternalServerError)


class BudgetedMessages:
    def __init__(self, client, budget, lock, model, ledger_path):
        self.client, self.budget, self.lock, self.model = client, budget, lock, model
        self.ledger_path = ledger_path

    async def create(self, **kwargs):
        if kwargs["model"] != self.model:
            raise RuntimeError(f"request model {kwargs['model']!r} != run model {self.model!r}")
        for attempt in range(2):
            async with self.lock:
                self.budget.take({"attempt": attempt})
            try:
                resp = await self.client.messages.create(**kwargs)
            except TRANSIENT as e:
                if attempt == 1:
                    raise
                await asyncio.sleep(5 if isinstance(e, anthropic.RateLimitError) else 1)
                continue
            if resp.model != self.model:
                raise RuntimeError(f"returned model {resp.model!r} != requested {self.model!r}")
            async with self.lock:
                with open(self.ledger_path, "a") as f:
                    f.write(json.dumps({
                        "t": datetime.now(timezone.utc).isoformat(), "id": resp.id,
                        "model": resp.model, "stop": resp.stop_reason,
                        "blocks": [b.type for b in resp.content],
                        "input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                    }) + "\n")
            return resp
        raise AssertionError("unreachable")


class BudgetedClient:
    def __init__(self, messages):
        self.messages = messages


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["prefix", "full"])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    run_dir = args.run_dir / f"{args.protocol}_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path, rej_path = run_dir / "labels.jsonl", run_dir / "rejects.jsonl"
    dialogues = [json.loads(l) for l in open(annotate.DATA / "dialogues_test.jsonl", encoding="utf-8")]
    if args.limit:
        dialogues = dialogues[: args.limit]

    done: dict = {}
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            done.setdefault(r["dialogue_id"], {})[r["turn"]] = r
    n_turns = {d["dialogue_id"]: sum(1 for t in d["turns"] if t["student"]) for d in dialogues}
    todo = [d for d in dialogues if len(done.get(d["dialogue_id"], {})) < n_turns[d["dialogue_id"]]]
    if args.protocol == "full":
        todo = [d for d in todo if d["dialogue_id"] not in done]

    budget, lock = Budget(run_dir, args.max_calls), asyncio.Lock()
    client = BudgetedClient(BudgetedMessages(make_client(), budget, lock, args.model,
                                             run_dir / "responses.jsonl"))
    print(f"{args.protocol}/test: {len(todo)}/{len(dialogues)} dialogues to do, "
          f"budget used {budget.used}/{args.max_calls}, model={args.model}", flush=True)
    sem, stats, errors = asyncio.Semaphore(args.concurrency), {}, []
    t0 = time.time()

    async def run_one(d):
        async with sem:
            try:
                if args.protocol == "prefix":
                    await annotate.annotate_dialogue(client, args.model, d, out_f, rej_f, lock, stats,
                                                     done.get(d["dialogue_id"], {}))
                else:
                    await annotate_leakage.annotate_dialogue(client, args.model, d, out_f, rej_f, lock, stats)
            except Exception as e:
                errors.append(d["dialogue_id"])
                print(f"[ERROR] {d['dialogue_id']}: {type(e).__name__}: {str(e)[:160]}", flush=True)

    with open(out_path, "a", encoding="utf-8") as out_f, open(rej_path, "a", encoding="utf-8") as rej_f:
        await asyncio.gather(*(run_one(d) for d in todo))

    summary = {"protocol": args.protocol, "model": args.model, "dialogues": len(dialogues),
               "errors": len(errors), "seconds": round(time.time() - t0), "budget_used": budget.used, **stats}
    with open(run_dir / "invocations.jsonl", "a") as f:
        f.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(), **summary}) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
