"""
simulation.py
Core discrete-event simulator for the ORIE 4580/5580 LLM serving project.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import heapq
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Query:
    qid: int
    arrival_time: float
    L: int                      # prompt length (tokens)
    B: int                      # decode tokens to generate

    # state
    prefill_done: bool = False
    tokens_prefilled: int = 0
    tokens_generated: int = 0

    # timestamps
    t_first_token: Optional[float] = None   # when prefill is fully done
    t_complete: Optional[float] = None      # when all decode tokens done

    # decode token completion times (for "true" TBT)
    decode_token_times: List[float] = field(default_factory=list)


@dataclass(order=True)
class Event:
    time: float
    priority: int
    event_type: str = field(compare=False)      # "arrival" or "batch_complete"
    payload: Any = field(compare=False, default=None)


@dataclass
class Batch:
    stage: str                   # "prefill" or "decode"
    tokens: int                  # total token load in this batch
    queries: List[Query]         # queries affected by this batch


# -----------------------------
# Scheduling policies
# -----------------------------

class BaseScheduler:
    """Schedulers must implement select_batch(...)->Batch|None."""
    def select_batch(
        self,
        t: float,
        prefill_queue: List[Query],
        decode_queue: List[Query],
        batch_cap: int,
    ) -> Optional[Batch]:
        raise NotImplementedError


class PrefillFirstScheduler(BaseScheduler):
    """
    Prefill-first (iteration-level):
    - If any prefill waiting, batch as many full prefills as fit (greedy FIFO).
    - Else decode: batch 1 decode token per query (greedy FIFO) up to batch_cap.
    """
    def select_batch(
        self,
        t: float,
        prefill_queue: List[Query],
        decode_queue: List[Query],
        batch_cap: int,
    ) -> Optional[Batch]:
        if prefill_queue:
            tokens = 0
            selected: List[Query] = []
            for q in prefill_queue:
                if tokens + q.L <= batch_cap:
                    selected.append(q)
                    tokens += q.L
                if tokens >= batch_cap:
                    break
            if not selected:
                q0 = prefill_queue[0]
                raise ValueError(
                    f"PrefillFirstScheduler: prompt length L={q0.L} exceeds batch_cap={batch_cap}. "
                    "Baseline requires L <= batch_cap (or implement chunked-prefill)."
                )
            return Batch(stage="prefill", tokens=tokens, queries=selected)

        if decode_queue:
            tokens = 0
            selected: List[Query] = []
            for q in decode_queue:
                if tokens + 1 <= batch_cap:
                    selected.append(q)
                    tokens += 1
                else:
                    break
            if not selected:
                return None
            return Batch(stage="decode", tokens=tokens, queries=selected)

        return None


class DecodeFirstScheduler(BaseScheduler):
    """
    Decode-first:
    - Always prioritize decode if any decode work exists.
    - Only run prefills when decode queue is empty.
    """
    def select_batch(
        self,
        t: float,
        prefill_queue: List[Query],
        decode_queue: List[Query],
        batch_cap: int,
    ) -> Optional[Batch]:
        if decode_queue:
            tokens = 0
            selected: List[Query] = []
            for q in decode_queue:
                if tokens + 1 <= batch_cap:
                    selected.append(q)
                    tokens += 1
                else:
                    break
            if not selected:
                return None
            return Batch(stage="decode", tokens=tokens, queries=selected)

        if prefill_queue:
            tokens = 0
            selected: List[Query] = []
            for q in prefill_queue:
                if tokens + q.L <= batch_cap:
                    selected.append(q)
                    tokens += q.L
                if tokens >= batch_cap:
                    break
            if not selected:
                q0 = prefill_queue[0]
                raise ValueError(
                    f"DecodeFirstScheduler: prompt length L={q0.L} exceeds batch_cap={batch_cap}. "
                    "Baseline requires L <= batch_cap (or implement chunked-prefill)."
                )
            return Batch(stage="prefill", tokens=tokens, queries=selected)

        return None


class HybridEveryNScheduler(BaseScheduler):
    """
    Simple hybrid to avoid starvation:
    - Serve decode whenever available EXCEPT every N batches (when prefill exists),
      run a prefill batch.
    """
    def __init__(self, n: int = 10):
        self.n = max(1, int(n))
        self._batch_counter = 0

    def select_batch(
        self,
        t: float,
        prefill_queue: List[Query],
        decode_queue: List[Query],
        batch_cap: int,
    ) -> Optional[Batch]:
        self._batch_counter += 1
        do_prefill = (self._batch_counter % self.n == 0) and bool(prefill_queue)

        if decode_queue and not do_prefill:
            tokens = 0
            selected: List[Query] = []
            for q in decode_queue:
                if tokens + 1 <= batch_cap:
                    selected.append(q)
                    tokens += 1
                else:
                    break
            if selected:
                return Batch(stage="decode", tokens=tokens, queries=selected)

        if prefill_queue:
            tokens = 0
            selected: List[Query] = []
            for q in prefill_queue:
                if tokens + q.L <= batch_cap:
                    selected.append(q)
                    tokens += q.L
                if tokens >= batch_cap:
                    break
            if not selected:
                q0 = prefill_queue[0]
                raise ValueError(
                    f"HybridEveryNScheduler: prompt length L={q0.L} exceeds batch_cap={batch_cap}. "
                    "Baseline requires L <= batch_cap (or implement chunked-prefill)."
                )
            return Batch(stage="prefill", tokens=tokens, queries=selected)

        if decode_queue:
            tokens = 0
            selected: List[Query] = []
            for q in decode_queue:
                if tokens + 1 <= batch_cap:
                    selected.append(q)
                    tokens += 1
                else:
                    break
            if selected:
                return Batch(stage="decode", tokens=tokens, queries=selected)

        return None


# -----------------------------
# Simulation core
# -----------------------------

class Simulation:
    def __init__(
        self,
        lam: float,
        batch_cap: int,
        c: float,
        a: float,
        b0: int,
        scheduler: BaseScheduler,
        rng_seed: int = 0,
        max_queries: int = 10_000,
        warmup_queries: int = 1_000,
        L_sampler=None,
        B_sampler=None,
        service_mode: str = "deterministic",   # NEW: "deterministic" or "stochastic"
        validation_mm1: bool = False,         # NEW: reduce to M/M/1-like single-stage system
    ):
        self.lam = float(lam)
        self.batch_cap = int(batch_cap)
        self.c = float(c)
        self.a = float(a)
        self.b0 = int(b0)
        self.scheduler = scheduler
        self.rng = np.random.default_rng(rng_seed)
        self.max_queries = int(max_queries)
        self.warmup_queries = int(warmup_queries)

        self._L_sampler = L_sampler
        self._B_sampler = B_sampler

        self.service_mode = str(service_mode).lower()
        if self.service_mode not in ("deterministic", "stochastic"):
            raise ValueError("service_mode must be 'deterministic' or 'stochastic'")

        self.validation_mm1 = bool(validation_mm1)

        self.t = 0.0
        self.event_queue: List[Event] = []

        self.prefill_queue: List[Query] = []
        self.decode_queue: List[Query] = []

        self.all_queries: List[Query] = []
        self.completed_queries: List[Query] = []
        self.next_qid: int = 0

        self.gpu_busy: bool = False
        self.t_busy: float = 0.0
        self.t_last_change: float = 0.0

        self.current_batch_size: int = 0
        self.area_num_in_system: float = 0.0
        self.last_event_time: float = 0.0

        # snapshots at warmup boundary (arrival of qid == warmup_queries)
        self.t_measure_start: Optional[float] = None
        self.area_at_measure_start: Optional[float] = None
        self.busy_at_measure_start: Optional[float] = None

    def sample_interarrival(self) -> float:
        return float(self.rng.exponential(1.0 / self.lam))

    def sample_L(self) -> int:
        """
        Baseline assumption: prefill fits in one batch.
        Enforce L <= batch_cap by clamping (Option 1).
        """
        if self._L_sampler is not None:
            L = int(self._L_sampler(self.rng))
        else:
            choices = [32, 128, 512]
            probs = [0.6, 0.3, 0.1]
            L = int(self.rng.choice(choices, p=probs))

        if L > self.batch_cap:
            L = self.batch_cap  # clamp
        return int(L)

    def sample_B(self) -> int:
        if self._B_sampler is not None:
            return int(self._B_sampler(self.rng))
        b = int(self.rng.geometric(p=0.15))
        return int(min(max(b, 1), 64))

    def service_time(self, tokens: int) -> float:
        """
        Batch service time for token load b=tokens.

        deterministic:
            S(b) = c + a * max(0, b - b0)

        stochastic (validation-friendly):
            C ~ Exp(mean=c)
            A ~ Exp(mean=a)
            S(b) = C + max(0, b - b0) * A
        """
        b = int(tokens)
        extra = max(0, b - self.b0)

        if self.service_mode == "deterministic":
            return float(self.c + self.a * extra)

        # stochastic: exponentials with specified means
        C = 0.0 if self.c <= 0 else float(self.rng.exponential(self.c))
        A = 0.0 if self.a <= 0 else float(self.rng.exponential(self.a))
        return float(C + extra * A)

    def _busy_time_so_far(self, t: Optional[float] = None) -> float:
        if t is None:
            t = self.t
        base = self.t_busy
        if self.gpu_busy:
            base += (t - self.t_last_change)
        return float(base)

    def _update_num_in_system_area(self, new_time: float) -> None:
        dt = new_time - self.last_event_time
        if dt > 0:
            num_in_system = len(self.prefill_queue) + len(self.decode_queue) + self.current_batch_size
            self.area_num_in_system += num_in_system * dt
            self.last_event_time = new_time

    def update_gpu_busy(self, is_busy: bool) -> None:
        if self.gpu_busy:
            self.t_busy += self.t - self.t_last_change
        self.gpu_busy = bool(is_busy)
        self.t_last_change = self.t

    def schedule_event(self, time: float, priority: int, event_type: str, payload=None) -> None:
        heapq.heappush(self.event_queue, Event(float(time), int(priority), event_type, payload))

    def init_first_arrival(self) -> None:
        t_arrival = self.t + self.sample_interarrival()
        self.schedule_event(t_arrival, 0, "arrival", None)

    def handle_arrival(self) -> None:
        if self.next_qid == self.warmup_queries:
            self.t_measure_start = self.t
            self.area_at_measure_start = self.area_num_in_system
            self.busy_at_measure_start = self._busy_time_so_far(self.t)

        # VALIDATION MODE: reduce to a single-stage system by creating jobs
        # that go directly to decode_queue with B=1 and no prefill.
        if self.validation_mm1:
            q = Query(
                qid=self.next_qid,
                arrival_time=self.t,
                L=0,
                B=1,
                prefill_done=True,
                tokens_prefilled=0,
                tokens_generated=0,
                t_first_token=self.t,   # immediately available
            )
            self.next_qid += 1
            self.all_queries.append(q)
            self.decode_queue.append(q)
        else:
            q = Query(
                qid=self.next_qid,
                arrival_time=self.t,
                L=self.sample_L(),
                B=self.sample_B(),
            )
            self.next_qid += 1
            self.all_queries.append(q)
            self.prefill_queue.append(q)

        if self.next_qid < self.max_queries:
            t_next = self.t + self.sample_interarrival()
            self.schedule_event(t_next, 0, "arrival", None)

        if not self.gpu_busy:
            self.try_start_batch()

    def handle_batch_complete(self, batch: Batch) -> None:
        self.update_gpu_busy(is_busy=False)
        self.current_batch_size = 0

        if batch.stage == "prefill":
            for q in batch.queries:
                q.tokens_prefilled = q.L
                q.prefill_done = True
                if q.t_first_token is None:
                    q.t_first_token = self.t
                self.decode_queue.append(q)

        elif batch.stage == "decode":
            for q in batch.queries:
                q.tokens_generated += 1
                q.decode_token_times.append(self.t)
                if q.tokens_generated >= q.B:
                    q.t_complete = self.t
                    self.completed_queries.append(q)
                else:
                    self.decode_queue.append(q)

        self.try_start_batch()

    def try_start_batch(self) -> None:
        batch = self.scheduler.select_batch(
            t=self.t,
            prefill_queue=self.prefill_queue,
            decode_queue=self.decode_queue,
            batch_cap=self.batch_cap,
        )
        if batch is None:
            return

        if batch.tokens > self.batch_cap:
            raise RuntimeError(
                f"Scheduler produced invalid batch: tokens={batch.tokens} > batch_cap={self.batch_cap}"
            )

        if batch.stage == "prefill":
            for q in batch.queries:
                if q in self.prefill_queue:
                    self.prefill_queue.remove(q)
        else:
            seen = set()
            for q in batch.queries:
                if q.qid in seen:
                    raise RuntimeError("Decode batch contained duplicate query.")
                seen.add(q.qid)
                if q in self.decode_queue:
                    self.decode_queue.remove(q)

        duration = self.service_time(batch.tokens)

        self.current_batch_size = len(batch.queries)
        self.update_gpu_busy(is_busy=True)

        t_done = self.t + duration
        self.schedule_event(t_done, 1, "batch_complete", batch)

    def run(self) -> Dict[str, float]:
        self.init_first_arrival()

        while self.event_queue and len(self.completed_queries) < self.max_queries:
            event = heapq.heappop(self.event_queue)

            self._update_num_in_system_area(event.time)
            self.t = event.time

            if event.event_type == "arrival":
                self.handle_arrival()
            elif event.event_type == "batch_complete":
                self.handle_batch_complete(event.payload)
            else:
                raise ValueError(f"Unknown event type: {event.event_type}")

        self.update_gpu_busy(is_busy=False)

        scored = [q for q in self.completed_queries if q.qid >= self.warmup_queries]
        return self._collect_metrics(scored)

    def export_traces(self) -> Dict[str, np.ndarray]:
        """
        Export raw traces for time-series plots (throughput / TTFT / TBT over time).

        Arrays are indexed by qid (0..num_arrivals-1). If a timestamp is not available,
        it will be NaN.
        """
        n = len(self.all_queries)
        arrival = np.full(n, np.nan, dtype=float)
        first_token = np.full(n, np.nan, dtype=float)
        complete = np.full(n, np.nan, dtype=float)
        ttft = np.full(n, np.nan, dtype=float)
        latency = np.full(n, np.nan, dtype=float)
        mean_tbt = np.full(n, np.nan, dtype=float)

        for q in self.all_queries:
            i = q.qid
            arrival[i] = q.arrival_time
            if q.t_first_token is not None:
                first_token[i] = q.t_first_token
                ttft[i] = q.t_first_token - q.arrival_time
            if q.t_complete is not None:
                complete[i] = q.t_complete
                latency[i] = q.t_complete - q.arrival_time

            if q.decode_token_times is not None and len(q.decode_token_times) >= 2:
                gaps = np.diff(np.array(q.decode_token_times, dtype=float))
                mean_tbt[i] = float(np.mean(gaps))

        return {
            "arrival_time": arrival,
            "first_token_time": first_token,
            "complete_time": complete,
            "ttft": ttft,
            "latency": latency,
            "mean_tbt": mean_tbt,
        }

    def _collect_metrics(self, queries: List[Query]) -> Dict[str, float]:
        if not queries:
            return {}

        ttfts: List[float] = []
        latencies: List[float] = []
        tbts: List[float] = []

        for q in queries:
            if q.t_first_token is None or q.t_complete is None:
                continue
            ttfts.append(q.t_first_token - q.arrival_time)
            latencies.append(q.t_complete - q.arrival_time)

            if len(q.decode_token_times) >= 2:
                gaps = np.diff(q.decode_token_times)
                tbts.append(float(np.mean(gaps)))

        def p95(x: Sequence[float]) -> float:
            return float(np.percentile(x, 95)) if len(x) else float("nan")

        if self.t_measure_start is None:
            t0, area0, busy0 = 0.0, 0.0, 0.0
        else:
            t0 = float(self.t_measure_start)
            area0 = float(self.area_at_measure_start or 0.0)
            busy0 = float(self.busy_at_measure_start or 0.0)

        T = float(self.t - t0)
        if T <= 0:
            throughput = util = mean_num_in_system = float("nan")
        else:
            throughput = len(queries) / T
            util = (self._busy_time_so_far(self.t) - busy0) / T
            mean_num_in_system = (self.area_num_in_system - area0) / T

        return {
            "num_queries": len(queries),
            "mean_ttft": float(np.mean(ttfts)) if ttfts else float("nan"),
            "p95_ttft": p95(ttfts),
            "mean_tbt": float(np.mean(tbts)) if tbts else float("nan"),
            "p95_tbt": p95(tbts),
            "mean_latency": float(np.mean(latencies)) if latencies else float("nan"),
            "p95_latency": p95(latencies),
            "throughput": throughput,
            "utilization": util,
            "mean_num_in_system": mean_num_in_system,
        }


# -----------------------------
# Experiment helpers
# -----------------------------

def run_one(
    lam: float,
    scheduler: BaseScheduler,
    *,
    batch_cap: int = 1024,
    c: float = 0.01,
    a: float = 0.0005,
    b0: int = 64,
    max_queries: int = 10_000,
    warmup_queries: int = 1_000,
    seed: int = 0,
    service_mode: str = "deterministic",
    validation_mm1: bool = False,
    return_traces: bool = False,
) -> Dict[str, float]:
    sim = Simulation(
        lam=lam,
        batch_cap=batch_cap,
        c=c,
        a=a,
        b0=b0,
        scheduler=scheduler,
        rng_seed=seed,
        max_queries=max_queries,
        warmup_queries=warmup_queries,
        service_mode=service_mode,
        validation_mm1=validation_mm1,
    )
    out = sim.run()
    out["lambda"] = float(lam)
    out["scheduler"] = type(scheduler).__name__
    out["seed"] = int(seed)
    out["service_mode"] = service_mode
    out["validation_mm1"] = bool(validation_mm1)

    if return_traces:
        out["_traces"] = sim.export_traces()

    return out


def run_grid(
    lambdas: Sequence[float],
    scheduler_factories: Sequence[Any],
    *,
    seeds: Sequence[int] = (0,),
    batch_cap: int = 1024,
    c: float = 0.01,
    a: float = 0.0005,
    b0: int = 64,
    max_queries: int = 10_000,
    warmup_queries: int = 1_000,
    service_mode: str = "deterministic",
    validation_mm1: bool = False,
) -> List[Dict[str, float]]:
    """
    Run a grid of experiments. Returns list of dict metrics (easy to convert to pandas.DataFrame).

    scheduler_factories can contain:
      - scheduler classes (e.g., PrefillFirstScheduler), or
      - zero-arg callables returning a scheduler (e.g., lambda: HybridEveryNScheduler(10)), or
      - an already-instantiated scheduler object.
    """
    rows: List[Dict[str, float]] = []

    def make_sched(sf: Any) -> BaseScheduler:
        if isinstance(sf, BaseScheduler):
            return sf
        if isinstance(sf, type):
            return sf()
        if callable(sf):
            return sf()
        raise TypeError("scheduler_factories must contain scheduler classes, callables, or instances.")

    for lam in lambdas:
        for sf in scheduler_factories:
            for seed in seeds:
                rows.append(
                    run_one(
                        lam,
                        make_sched(sf),
                        batch_cap=batch_cap,
                        c=c,
                        a=a,
                        b0=b0,
                        max_queries=max_queries,
                        warmup_queries=warmup_queries,
                        seed=seed,
                        service_mode=service_mode,
                        validation_mm1=validation_mm1,
                    )
                )
    return rows