# ORIE 4580/5580 – LLM Query Serving System Simulation

This project implements a discrete-event stochastic simulation of a large-scale LLM query serving system.
The simulator models:

* Poisson query arrivals,
* Two-stage LLM inference (prefill → decode),
* GPU iteration-level batching with a token capacity limit, and
* Multiple scheduling policies (prefill-first, decode-first, hybrid).

The goal is to evaluate how scheduling affects throughput, latency (TTFT, TBT, end-to-end), queueing dynamics, and GPU utilization across different load levels. The final notebook presents model validation, system behavior, and comparative policy analysis.

---

## File Structure

* **main_report.ipynb** – Final project notebook containing the written report, model explanation, validation experiments, plots, and analysis.
* **simulation.py** – Core discrete-event simulator and all scheduling policies (Prefill-First, Decode-First, Hybrid), plus experiment helpers and an M/M/1 validation function.
* **README.md** – This file.

---

## How to Run

1. Open a fresh Jupyter Notebook or Google Colab session.
2. Upload **main_report.ipynb** and **simulation.py** into the same working directory.
3. Run **main_report.ipynb** from top to bottom.

All tables and plots are generated on the fly by calling the functions defined in `simulation.py`.
No pre-generated datasets are required.

---

## Dependencies

This project uses only standard scientific Python packages:

* **Python 3.9+**
* **numpy**
* **pandas**
* **matplotlib**

These are automatically available in Google Colab and common scientific Python environments.

---

## Modeling Assumptions

### **Arrivals**

* Queries arrive according to a **Poisson process** with rate ( \lambda ).
* Interarrival times are exponential.

### **LLM Inference Structure**

Each query has two stages:

1. **Prefill phase**

   * The entire prompt (L tokens) is processed in a single batch.
   * The moment prefill ends is recorded as **TTFT (time to first token)**.

2. **Decode phase**

   * Output tokens (B tokens) are generated sequentially, one per iteration.
   * Per-token completion timestamps are recorded to compute **TBT (time between tokens)**.

### **GPU Batching Model**

* A single GPU worker processes **iteration-level batches**.
* A batch can include multiple queries up to a global **token capacity** (`batch_cap`).
* Batch service time uses the deterministic affine model:

  $$
  S(b) = c + a \cdot \max(0,\, b - b_0)
  $$

### **Scheduling Policies**

`simulation.py` includes:

* **Prefill-First**: always prioritize prefill work
* **Decode-First**: always prioritize decode tokens
* **HybridEveryN**: avoid starvation by periodically forcing a prefill batch

### **Steady-State Measurement**

* A warm-up period (`warmup_queries`) is discarded.
* Metrics are computed only from queries after warm-up to approximate long-run steady state.

---

## Notes

* All simulator logic, event handling, batching rules, and scheduling policies are implemented in **simulation.py**.
* `main_report.ipynb` handles:

  * Model explanation and diagrams
  * M/M/1 validation experiment
  * Throughput, latency, TTFT/TBT, and utilization plots
  * Scheduling policy comparisons
  * Conclusions and interpretation
* The simulator can be easily extended to support:

  * Chunked prefill,
  * Token-parallel GPUs,
  * Multi-GPU scheduling,
  * More advanced service-time distributions.
