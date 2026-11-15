"""GPU state probe: three quick measurements that show whether the GPU is in a normal or slow state.
1. Memory bandwidth: copy a 512 MiB buffer (reads + writes 1 GiB) - the 'memory wall' itself.
2. fp16 matrix-vector product at a Qwen3-1.7B MLP shape (6144x2048 weights) - what decode spends its time on.
3. Launch overhead: 2000 tiny kernels back to back.
4. Shader clock: one kernel reads the shader cycle counter and the constant 100 MHz wall clock before and after a busy
   loop; cycles / wall time = the clock the GPU cores actually ran at (MHz).
5. Sustained load: 12 seconds of GEMVs, reporting GB/s and shader clock per 2-second window (shows thermal/power throttling).
Each is timed 3 times with warm-up; medians are printed. Read-only, changes nothing.
Run: .venv-gpu\\Scripts\\python.exe -u gpu_state.py
"""
import statistics
import time

import torch


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


CLK_SRC = r"""
extern "C" __global__ void clk(unsigned long long* out, int n) {
    unsigned long long c0 = __builtin_readcyclecounter();
    unsigned long long w0 = wall_clock64();
    float a = 1.0f + threadIdx.x * 1e-7f;
    for (int i = 0; i < n; ++i) a = a * 0.9999999f + 1e-7f;
    unsigned long long c1 = __builtin_readcyclecounter();
    unsigned long long w1 = wall_clock64();
    if (threadIdx.x == 0) { out[0] = c1 - c0; out[1] = w1 - w0; out[2] = (unsigned long long)(a * 1000.0f); }
}
"""


def clock_mhz(k, buf, n=2_000_000):
    k(grid=(1, 1, 1), block=(64, 1, 1), args=[buf, n])
    torch.cuda.synchronize()
    c, w = int(buf[0]), int(buf[1])
    return c / (w / 100.0) if w > 0 else float("nan")       # wall clock = 100 MHz -> w/100 microseconds


def main():
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)
    torch.manual_seed(0)
    a = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)
    b = torch.empty_like(a)
    t = timed(lambda: b.copy_(a), 20)
    print(f"STATE bandwidth: {2 * a.numel() / t / 1e9:.0f} GB/s (copy 512 MiB; vendor peak 640 GB/s)", flush=True)
    W = torch.randn(6144, 2048, dtype=torch.float16, device=dev)
    x = torch.randn(2048, 1, dtype=torch.float16, device=dev)
    t = timed(lambda: [W @ x for _ in range(50)], 10) / 50
    print(f"STATE fp16 GEMV 6144x2048: {t * 1e6:.1f} us = {W.numel() * 2 / t / 1e9:.0f} GB/s effective", flush=True)
    s = torch.zeros(1, device=dev)
    t = timed(lambda: [s.add_(1) for _ in range(2000)], 5) / 2000
    print(f"STATE launch: {t * 1e6:.2f} us per tiny kernel", flush=True)
    try:
        k = torch.cuda._compile_kernel(CLK_SRC, "clk")
        buf = torch.zeros(4, dtype=torch.int64, device=dev)
        idle = clock_mhz(k, buf)
        busy = [W @ x for _ in range(2000)]
        loaded = clock_mhz(k, buf)
        print(f"STATE shader clock: {idle:.0f} MHz right after a pause | {loaded:.0f} MHz right after load", flush=True)
        t_end = time.perf_counter() + 12
        win = 0
        while time.perf_counter() < t_end:
            t0 = time.perf_counter(); n = 0
            while time.perf_counter() - t0 < 2:
                for _ in range(200):
                    W @ x
                torch.cuda.synchronize(); n += 200
            dt = time.perf_counter() - t0
            win += 1
            print(f"STATE sustained window {win}: {n * W.numel() * 2 / dt / 1e9:.0f} GB/s GEMV | clock "
                  f"{clock_mhz(k, buf):.0f} MHz", flush=True)
        del busy
    except Exception as ex:
        print(f"STATE clock probe failed: {str(ex)[:300]}", flush=True)
    del a, b, W


if __name__ == "__main__":
    main()
