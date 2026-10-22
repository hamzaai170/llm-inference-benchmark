#!/usr/bin/env python3
"""
Energy- and latency-aware benchmarking of open-weight LLMs under different
numeric precisions, for local ("sovereign") deployment.

For every (model, precision) pair it measures, in a fresh subprocess so that
memory numbers are not contaminated by earlier runs:
  - model size on disk (MB, serialised state dict)
  - load time (s)
  - time to first token (s) and generation throughput (tokens/s)
  - peak memory (MB; GPU allocated memory on CUDA, process RSS on CPU)
  - estimated energy (Wh) and CO2e (g) via CodeCarbon, reported per 1,000 tokens
  - perplexity on a slice of WikiText-2 (a quality proxy, lower is better)

Usage (see README.md):
  python llm_inference_benchmark.py --models Qwen/Qwen2.5-0.5B-Instruct --precisions fp32,int8dyn
"""
import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import time

# Short prompts from the three application domains named in the PhD posting.
PROMPTS = [
    # digital industries
    "Explain predictive maintenance for a factory production line in three sentences.",
    "List three risks of sending confidential process data to an external AI provider.",
    "Summarise why a small company might prefer running a language model on its own servers.",
    # automotive
    "Describe how an AI assistant could help a service technician diagnose a vehicle fault.",
    "Explain in three sentences what over-the-air software updates mean for car safety.",
    "Give two reasons why in-vehicle AI should work without a cloud connection.",
    # healthcare
    "Explain why patient records need extra protection when used with AI tools.",
    "Describe in three sentences how an AI tool could help a nurse draft a shift handover note.",
    "List two trade-offs of running a clinical language model on a hospital's own hardware.",
]

ALL_PRECISIONS_CPU = ["fp32", "bf16", "int8dyn", "int8sel"]
ALL_PRECISIONS_GPU = ["fp32", "fp16", "bnb8", "bnb4"]


# ----------------------------------------------------------------------------
# Worker: runs ONE (model, precision) configuration and prints a JSON line.
# ----------------------------------------------------------------------------
def peak_memory_mb(device):
    if device == "cuda":
        import torch
        return torch.cuda.max_memory_allocated() / 1e6
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak:
            return peak / 1e6
    except Exception:
        pass
    try:
        import resource  # Linux / macOS
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1e6 if sys.platform == "darwin" else 1e3)
    except Exception:
        return float("nan")


def serialised_size_mb(model):
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def perplexity(model, tok, device, windows, ctx=512):
    try:
        from datasets import load_dataset
    except Exception:
        return float("nan")
    import torch
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])[:300000]
    ids = tok(text, return_tensors="pt").input_ids
    nll, ntok = 0.0, 0
    for i in range(windows):
        chunk = ids[:, i * ctx:(i + 1) * ctx]
        if chunk.shape[1] < ctx:
            break
        chunk = chunk.to(device)
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        n = chunk.shape[1] - 1
        nll += float(loss.float().item()) * n
        ntok += n
    return math.exp(nll / ntok) if ntok else float("nan")


def run_worker(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = a.precision
    want_cuda = torch.cuda.is_available() and a.device in ("auto", "cuda")
    if p in ("int8dyn", "int8sel") or a.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if want_cuda else "cpu"
    if p in ("fp16", "bnb8", "bnb4") and device != "cuda":
        raise RuntimeError(f"precision {p} needs a CUDA GPU")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.perf_counter()
    if p in ("fp32", "int8dyn", "int8sel"):
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    elif p == "bf16":
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    elif p == "fp16":
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16)
    elif p in ("bnb8", "bnb4"):
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_8bit=(p == "bnb8"), load_in_4bit=(p == "bnb4"))
        model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=q, device_map="auto")
    else:
        raise RuntimeError(f"unknown precision {p}")
    if p == "int8dyn":
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    if p == "int8sel":
        # Selective int8: quantise only the Linear layers inside the transformer blocks.
        # The output head (lm_head, tied to the embeddings in Qwen2.5) stays in fp32.
        names = {n for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
        model = torch.ao.quantization.quantize_dynamic(model, names, dtype=torch.qint8)
    if p not in ("bnb8", "bnb4"):
        model = model.to(device)
    model.eval()
    load_s = time.perf_counter() - t0
    size_mb = serialised_size_mb(model)

    def encode(prompt):
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        return tok(text, return_tensors="pt").to(device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def generate(inputs, n_new):
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=n_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id)

    prompts = PROMPTS[: a.n_prompts]
    generate(encode(prompts[0]), 4)  # warm-up
    sync()

    # time to first token
    ttfts = []
    for pr in prompts:
        inputs = encode(pr)
        sync(); t = time.perf_counter()
        generate(inputs, 1)
        sync(); ttfts.append(time.perf_counter() - t)

    # energy tracking around the full generation loop
    tracker = None
    try:
        from codecarbon import EmissionsTracker
        try:
            tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False,
                                       log_level="error", allow_multiple_runs=True)
        except TypeError:
            tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level="error")
        tracker.start()
    except Exception:
        tracker = None

    total_new, total_time = 0, 0.0
    for pr in prompts:
        inputs = encode(pr)
        n_in = inputs["input_ids"].shape[1]
        sync(); t = time.perf_counter()
        out = generate(inputs, a.max_new_tokens)
        sync(); total_time += time.perf_counter() - t
        total_new += int(out.shape[1] - n_in)

    energy_wh, co2_g = float("nan"), float("nan")
    if tracker is not None:
        try:
            kg = tracker.stop()
            data = getattr(tracker, "final_emissions_data", None)
            if data is not None:
                energy_wh = float(data.energy_consumed) * 1000.0
            if kg is not None:
                co2_g = float(kg) * 1000.0
        except Exception:
            pass

    ppl = perplexity(model, tok, device, a.ppl_windows) if a.ppl_windows > 0 else float("nan")
    res = {
        "model": a.model, "precision": p, "device": device,
        "size_mb": round(size_mb, 1), "load_s": round(load_s, 2),
        "ttft_s": round(sum(ttfts) / len(ttfts), 4),
        "tokens_per_s": round(total_new / total_time, 2) if total_time else float("nan"),
        "generated_tokens": total_new,
        "peak_mem_mb": round(peak_memory_mb(device), 1),
        "energy_wh_total": energy_wh, "co2_g_total": co2_g,
        "energy_wh_per_1k_tokens": (energy_wh / total_new * 1000.0) if total_new and energy_wh == energy_wh else float("nan"),
        "perplexity_wikitext2": round(ppl, 3) if ppl == ppl else float("nan"),
    }
    print("RESULT_JSON:" + json.dumps(res))


# ----------------------------------------------------------------------------
# Orchestrator: loops over configurations, one subprocess each, then reports.
# ----------------------------------------------------------------------------
FIELDS = ["model", "precision", "device", "size_mb", "load_s", "ttft_s", "tokens_per_s",
          "generated_tokens", "peak_mem_mb", "energy_wh_total", "co2_g_total",
          "energy_wh_per_1k_tokens", "perplexity_wikitext2"]


def short(m):
    return m.split("/")[-1]


def write_report(rows, outdir):
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) and v == v else ("n/a" if isinstance(v, float) else str(v))

    heads = ["Model", "Precision", "Size MB", "Peak mem MB", "TTFT s", "Tokens/s",
             "Wh / 1k tok", "WikiText-2 PPL"]
    keys = ["model", "precision", "size_mb", "peak_mem_mb", "ttft_s", "tokens_per_s",
            "energy_wh_per_1k_tokens", "perplexity_wikitext2"]
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for r in rows:
        vals = [short(r["model"]) if k == "model" else fmt(r[k]) for k in keys]
        lines.append("| " + " | ".join(vals) + " |")
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping plots")
        return
    labels = [f"{short(r['model'])}\n{r['precision']}" for r in rows]
    for key, title, fname in [("tokens_per_s", "Generation throughput (tokens/s, higher is better)", "throughput.png"),
                              ("peak_mem_mb", "Peak memory (MB, lower is better)", "memory.png"),
                              ("energy_wh_per_1k_tokens", "Estimated energy per 1,000 tokens (Wh, lower is better)", "energy.png")]:
        vals = [r[key] if r[key] == r[key] else 0 for r in rows]
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(rows)), 4))
        ax.bar(range(len(rows)), vals)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)
    pts = [r for r in rows if r["energy_wh_per_1k_tokens"] == r["energy_wh_per_1k_tokens"]
           and r["perplexity_wikitext2"] == r["perplexity_wikitext2"]]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for r in pts:
            ax.scatter(r["energy_wh_per_1k_tokens"], r["perplexity_wikitext2"])
            ax.annotate(f"{short(r['model'])} {r['precision']}", (r["energy_wh_per_1k_tokens"], r["perplexity_wikitext2"]), fontsize=7)
        ax.set_xlabel("Estimated energy per 1,000 tokens (Wh)")
        ax.set_ylabel("WikiText-2 perplexity")
        ax.set_title("Efficiency vs quality trade-off (bottom-left is best)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "tradeoff.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="Qwen/Qwen2.5-0.5B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--precisions", default="", help="comma list; default depends on device")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--n-prompts", type=int, default=9)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--ppl-windows", type=int, default=20, help="512-token windows of WikiText-2; 0 to skip")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--precision", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        run_worker(a)
        return

    if a.precisions:
        precisions = a.precisions.split(",")
    else:
        import torch
        use_gpu = torch.cuda.is_available() and a.device != "cpu"
        precisions = ALL_PRECISIONS_GPU if use_gpu else ALL_PRECISIONS_CPU
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for m in a.models.split(","):
        for p in precisions:
            print(f"\n=== {m} | {p} ===", flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--model", m, "--precision", p,
                   "--device", a.device, "--n-prompts", str(a.n_prompts),
                   "--max-new-tokens", str(a.max_new_tokens), "--ppl-windows", str(a.ppl_windows)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
            if proc.returncode != 0 or line is None:
                print("FAILED:", (proc.stderr or proc.stdout)[-600:])
                continue
            rows.append(json.loads(line[len("RESULT_JSON:"):]))
            print({k: rows[-1][k] for k in ("tokens_per_s", "peak_mem_mb", "energy_wh_per_1k_tokens", "perplexity_wikitext2")})
    if rows:
        write_report(rows, a.outdir)
        print(f"\nSaved results to {a.outdir}/")
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()
