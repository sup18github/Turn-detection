import time
import numpy as np

def benchmark_inference_latency(model_fn, sample_generator, num_runs: int = 50) -> dict:
    """
    Measures Median Latency, P95 Latency, and Real-Time Factor (RTF).
    """
    latencies_ms = []
    audio_durations_s = []

    for _ in range(num_runs):
        audio_chunk, audio_duration = sample_generator()
        audio_durations_s.append(audio_duration)

        t0 = time.perf_counter()
        _ = model_fn(audio_chunk)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    median_latency_ms = float(np.median(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))

    total_processing_s = np.sum(latencies_ms) / 1000.0
    total_audio_s = np.sum(audio_durations_s)
    rtf = float(total_processing_s / total_audio_s) if total_audio_s > 0 else 0.0

    return {
        "median_latency_ms": median_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "real_time_factor": rtf,
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms))
    }
