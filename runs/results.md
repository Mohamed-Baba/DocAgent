# Results (2026-11-14 13:26, model qwen3-vl:8b-instruct)

### rag_dense

| category | n | accuracy [95% CI] | cit. precision | evidence seen | correct abst. | false abst. | calls | tokens | latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| simple | 4 | 0.25 [0.00, 0.75] | 0.33 | 0.25 | - | 0.25 | 1.0 | 1606 | 2.6 |
| all | 11 | 0.45 [0.18, 0.73] | 0.67 | 0.50 | 1.00 | 0.25 | 1.0 | 1579 | 1.5 |
| multistep | 4 | 0.25 [0.00, 0.75] | 1.00 | 0.75 | - | 0.25 | 1.0 | 1477 | 1.0 |
| unanswerable | 3 | 1.00 [1.00, 1.00] | - | - | 1.00 | - | 1.0 | 1680 | 0.8 |

### rag_hybrid

| category | n | accuracy [95% CI] | cit. precision | evidence seen | correct abst. | false abst. | calls | tokens | latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| simple | 4 | 0.75 [0.25, 1.00] | 1.00 | 1.00 | - | 0.25 | 1.0 | 1754 | 1.6 |
| all | 11 | 0.55 [0.27, 0.82] | 1.00 | 0.94 | 1.00 | 0.12 | 1.0 | 1762 | 1.8 |
| multistep | 4 | 0.00 [0.00, 0.00] | 1.00 | 0.88 | - | 0.00 | 1.0 | 1732 | 2.0 |
| unanswerable | 3 | 1.00 [1.00, 1.00] | - | - | 1.00 | - | 1.0 | 1812 | 2.0 |

### agent

| category | n | accuracy [95% CI] | cit. precision | evidence seen | correct abst. | false abst. | calls | tokens | latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| simple | 4 | 0.50 [0.00, 1.00] | 0.67 | 1.00 | - | 0.25 | 3.5 | 7657 | 11.9 |
| all | 11 | 0.64 [0.36, 0.91] | 0.71 | 0.81 | 0.67 | 0.12 | 4.2 | 12614 | 32.8 |
| multistep | 4 | 0.75 [0.25, 1.00] | 1.00 | 0.62 | - | 0.00 | 3.5 | 5777 | 30.0 |
| unanswerable | 3 | 0.67 [0.00, 1.00] | 0.00 | - | 0.67 | - | 6.0 | 28338 | 64.4 |

### agent_verified

| category | n | accuracy [95% CI] | cit. precision | evidence seen | correct abst. | false abst. | calls | tokens | latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| simple | 4 | 0.25 [0.00, 0.75] | 0.50 | 1.00 | - | 0.50 | 4.2 | 10173 | 23.4 |
| all | 11 | 0.55 [0.27, 0.82] | 0.67 | 0.81 | 0.67 | 0.38 | 5.4 | 13409 | 42.5 |
| multistep | 4 | 0.75 [0.25, 1.00] | 1.00 | 0.62 | - | 0.25 | 6.0 | 9680 | 36.0 |
| unanswerable | 3 | 0.67 [0.00, 1.00] | 0.00 | - | 0.67 | - | 6.0 | 22696 | 76.7 |

### Paired comparisons (exact McNemar, Holm-corrected at 0.05)

| pair | first right only | second right only | p | rejected |
|---|---|---|---|---|
| rag_hybrid vs rag_dense | 2 | 1 | 1.000 | False |
| agent vs rag_hybrid | 3 | 2 | 1.000 | False |
| agent_verified vs agent | 1 | 2 | 1.000 | False |
| agent_verified vs rag_hybrid | 3 | 3 | 1.000 | False |

VRAM used by the LLM: 7.86 GB