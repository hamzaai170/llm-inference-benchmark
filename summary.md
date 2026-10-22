| Model | Precision | Size MB | Peak mem MB | TTFT s | Tokens/s | Wh / 1k tok | WikiText-2 PPL |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | fp32 | 1976.20 | 4456.90 | 1.49 | 4.11 | 0.92 | 21.68 |
| Qwen2.5-0.5B-Instruct | bf16 | 988.20 | 2527.70 | 4.01 | 3.46 | 1.06 | 21.73 |
| Qwen2.5-0.5B-Instruct | int8dyn | 1039.00 | 5440.30 | 0.60 | 9.49 | 0.39 | 174.78 |
| TinyLlama-1.1B-Chat-v1.0 | fp32 | 4400.30 | 9113.50 | 1.97 | 2.39 | 1.53 | 11.79 |
| TinyLlama-1.1B-Chat-v1.0 | bf16 | 2200.20 | 4823.80 | 7.89 | 1.77 | 2.09 | 11.79 |
| TinyLlama-1.1B-Chat-v1.0 | int8dyn | 1297.10 | 10527.90 | 0.93 | 6.08 | 0.60 | 77.79 |
