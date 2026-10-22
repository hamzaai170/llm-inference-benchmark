# Energy- and Latency-Aware Benchmarking of Open-Weight LLMs under Different Numeric Precisions

A small, reproducible pilot benchmark of how numeric precision (fp32, bf16, dynamic int8) affects the **speed, memory, estimated energy and output quality** of open-weight language models running locally on CPU. It was built as a first step towards the questions of sustainable, locally deployable ("sovereign") generative AI.

## What is measured
Each (model, precision) pair runs in a **fresh subprocess**, so memory numbers are not contaminated by earlier runs:

| Metric | How |
|---|---|
| Model size (MB) | serialised `state_dict` |
| Time to first token (s) | mean over prompts, 1 generated token |
| Throughput (tokens/s) | greedy decoding, total new tokens / wall time |
| Peak memory (MB) | process RSS (CPU) or max allocated (CUDA) |
| Energy (Wh per 1,000 tokens) | [CodeCarbon](https://github.com/mlco2/codecarbon) estimate around the generation loop |
| Quality | perplexity on WikiText-2 test text (ten 512-token windows) |

Prompts: 9 short prompts from the three application domains in the PhD posting (digital industries, automotive, healthcare), 48 new tokens each.

## Results (CPU, 2-core Google Colab instance, single run)

| Model | Precision | Size MB | Peak mem MB | TTFT s | Tokens/s | Wh / 1k tok | WikiText-2 PPL |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | fp32 | 1976.20 | 4456.90 | 1.49 | 4.11 | 0.92 | 21.68 |
| Qwen2.5-0.5B-Instruct | bf16 | 988.20 | 2527.70 | 4.01 | 3.46 | 1.06 | 21.73 |
| Qwen2.5-0.5B-Instruct | int8dyn | 1039.00 | 5440.30 | 0.60 | 9.49 | 0.39 | 174.78 |
| TinyLlama-1.1B-Chat-v1.0 | fp32 | 4400.30 | 9113.50 | 1.97 | 2.39 | 1.53 | 11.79 |
| TinyLlama-1.1B-Chat-v1.0 | bf16 | 2200.20 | 4823.80 | 7.89 | 1.77 | 2.09 | 11.79 |
| TinyLlama-1.1B-Chat-v1.0 | int8dyn | 1297.10 | 10527.90 | 0.93 | 6.08 | 0.60 | 77.79 |

Charts: `results/throughput.png`, `results/energy.png`, `results/size.png`, `results/memory.png`, `results/tradeoff.png`. Raw data: `results/results.csv`.

## Findings (relative to fp32 of the same model)
- **Dynamic int8 is much faster and cheaper, but loses a lot of quality.** Throughput 2.3x (Qwen) and 2.5x (TinyLlama); estimated energy per 1,000 tokens down 58% and 61%; model size down 47% and 71%. Perplexity rises from 21.7 to 174.8 (Qwen) and from 11.8 to 77.8 (TinyLlama).
- **bf16 halves model size and cuts peak memory by 43% (Qwen) and 47% (TinyLlama) with no perplexity change, but was slower** (16% and 26% fewer tokens/s) and used more estimated energy on this CPU, which has no native bf16 acceleration.
- **Efficiency claims need a quality measure beside them.** The most energy-efficient configurations here are also the lowest-quality ones.

## Limitations (please read)
- Small pilot: 9 prompts, 48 new tokens, ten perplexity windows, **one run per configuration**, no repeats or confidence intervals.
- Shared cloud CPU. Timings and energy can vary between runs and instances.
- **Energy is a CodeCarbon software estimate, not a power-meter measurement.** Use it to compare rows with each other, not as an absolute figure.
- **Int8 peak memory is misleading.** The script loads the fp32 model first and then quantises it, so the peak includes the full-size copy. Use model size for int8 memory claims.
- The int8dyn setting quantises *all* `Linear` layers, including the output head (`lm_head`), which is likely a main cause of the large perplexity increase. A selective variant (`int8sel`, transformer blocks only) is included in the script; see below.
- Perplexity on WikiText-2 is a rough quality proxy and says little about instruction-following or domain performance.

## Reproduce
```bash
pip install torch transformers datasets codecarbon psutil matplotlib
python llm_inference_benchmark.py --device cpu \
  --models Qwen/Qwen2.5-0.5B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --precisions fp32,bf16,int8dyn --n-prompts 9 --max-new-tokens 48 --ppl-windows 10 --outdir results
```
Selective int8 follow-up (quantises only the transformer-block `Linear` layers, keeps the output head in fp32):
```bash
python llm_inference_benchmark.py --device cpu --models Qwen/Qwen2.5-0.5B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --precisions int8sel --n-prompts 9 --max-new-tokens 48 --ppl-windows 10 --outdir results_int8sel
```
Expect roughly 15 to 70 minutes on a free Colab CPU depending on the model list.

## Next steps
Repeat runs with confidence intervals; measure power with hardware counters instead of an estimate; add GPU precisions (fp16, 8-bit, 4-bit); evaluate task quality on domain prompts instead of perplexity alone.
