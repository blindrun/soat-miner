#!/usr/bin/env python3
"""
soat-miner-guard - decides when it is acceptable to mine.

The miner itself knows nothing about this. The guard starts and stops the
systemd unit, and manages the GPU power limit, based on four questions:

  1. Is anything actually using the GPU for real work?
     Ollama and ComfyUI both expose precise "am I busy" endpoints, which beats
     polling GPU utilisation: ComfyUI holds ~400 MB of VRAM while completely
     idle, so a VRAM or utilisation threshold either false-positives forever
     or reacts too late. /api/ps and /queue answer exactly the right question.

  2. Is it within the allowed hours?
     Summer default avoids the heat of the day. One config line to change for
     winter - see MINE_WINDOWS.

  3. Is the GPU too hot?
     A backstop independent of the clock, in case the room is hot anyway.

  4. Has it been idle long enough to be worth restarting?
     Rebuilding the 7 GB dataset costs ~4 s, so the guard waits for a short
     settle period before resuming rather than thrashing on every brief gap
     between two prompts.

Stopping is immediate and unconditional; only starting is delayed. Losing a
few seconds of mining is free, making an interactive prompt wait is not.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, time as dtime

# ----------------------------------------------------------------- config --
CONFIG_PATH = os.environ.get("SOAT_MINER_CONFIG", "/etc/soat-miner/guard.conf")

DEFAULTS = {
    # Comma-separated HH:MM-HH:MM windows in LOCAL time. Windows may wrap
    # midnight. Summer default: overnight only, avoiding the heat of the day.
    # Winter: widen to "00:00-23:59" for all day.
    "MINE_WINDOWS": "20:00-10:00",
    "POLL_SECONDS": "1",
    # Consecutive idle polls required before (re)starting.
    "IDLE_SETTLE_POLLS": "5",
    "GPU_TEMP_MAX_C": "70",
    "GPU_INDEX": "0",
    "POWER_LIMIT_MINING_W": "183",
    "POWER_LIMIT_IDLE_W": "450",
    "MINER_UNIT": "soat-miner.service",
    "OLLAMA_URL": "http://127.0.0.1:11434",
    "COMFYUI_URL": "http://127.0.0.1:8188",
    # Any other process holding more than this much VRAM counts as busy.
    # ComfyUI idles around 400 MB, so the default sits above that.
    "FOREIGN_VRAM_MB": "500",
    # Processes that mean "GPU work is starting" the moment they appear, at
    # ANY vram size. This matters: qwen3:8k needs 21.9 GB and the miner holds
    # 7.3 GB on a 24 GB card, so they cannot coexist - waiting for a vram
    # threshold to trip means reacting after the allocation has already
    # failed. Matched as substrings of the process path.
    "BUSY_PROCESS_NAMES": "llama-server,ollama",
    "DRY_RUN": "false",
}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    for k, v in os.environ.items():
        if k in cfg:
            cfg[k] = v
    return cfg


def log(msg):
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


# ------------------------------------------------------------- detectors --
def http_json(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def ollama_busy(cfg):
    """True if any model is currently loaded/running (i.e. Qwen was asked)."""
    d = http_json(f"{cfg['OLLAMA_URL']}/api/ps")
    if d is None:
        return False  # service down means not busy
    return len(d.get("models", [])) > 0


def comfyui_busy(cfg):
    """True if anything is running or queued."""
    d = http_json(f"{cfg['COMFYUI_URL']}/queue")
    if d is None:
        return False
    return bool(d.get("queue_running")) or bool(d.get("queue_pending"))


def nvidia_query(fields, cfg):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
             "-i", cfg["GPU_INDEX"]],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return [x.strip() for x in out.stdout.strip().split(",")]
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_temp(cfg):
    v = nvidia_query("temperature.gpu", cfg)
    try:
        return int(v[0]) if v else None
    except (ValueError, IndexError):
        return None


def foreign_gpu_processes(cfg, miner_pids):
    """
    Any non-miner compute process holding significant VRAM.
    Catches work that does not go through Ollama or ComfyUI at all - a bare
    PyTorch script, a training run, someone's notebook.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits", "-i", cfg["GPU_INDEX"]],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
    except (OSError, subprocess.SubprocessError):
        return []

    threshold = int(cfg["FOREIGN_VRAM_MB"])
    names = [n.strip().lower() for n in cfg["BUSY_PROCESS_NAMES"].split(",")
             if n.strip()]
    busy = []
    for line in out.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid, name, mem = int(parts[0]), parts[1], int(parts[2])
        except (ValueError, IndexError):
            continue
        if pid in miner_pids:
            continue
        lname = name.lower()
        # Name match fires immediately, before the allocation grows.
        if any(n in lname for n in names):
            busy.append((pid, f"{os.path.basename(name)} starting", mem))
        elif mem >= threshold:
            busy.append((pid, os.path.basename(name), mem))
    return busy


# ---------------------------------------------------------------- window --
def parse_windows(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part or "-" not in part:
            continue
        a, b = part.split("-", 1)
        try:
            ah, am = (int(x) for x in a.split(":"))
            bh, bm = (int(x) for x in b.split(":"))
        except ValueError:
            continue
        out.append((dtime(ah, am), dtime(bh, bm)))
    return out


def in_window(now, windows):
    if not windows:
        return True
    t = now.time()
    for start, end in windows:
        if start <= end:
            if start <= t <= end:
                return True
        else:  # wraps midnight
            if t >= start or t <= end:
                return True
    return False


# ---------------------------------------------------------------- systemd --
def unit_active(unit):
    r = subprocess.run(["systemctl", "is-active", "--quiet", unit])
    return r.returncode == 0


def unit_pids(unit):
    try:
        out = subprocess.run(
            ["systemctl", "show", "-p", "MainPID", "--value", unit],
            capture_output=True, text=True, timeout=10,
        )
        pid = int(out.stdout.strip() or 0)
        if pid <= 0:
            return set()
        pids = {pid}
        # include children, since the miner may fork
        try:
            ch = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True,
                                text=True, timeout=10)
            pids |= {int(x) for x in ch.stdout.split()}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return pids
    except (OSError, subprocess.SubprocessError, ValueError):
        return set()


def set_power_limit(watts, cfg):
    if cfg["DRY_RUN"].lower() == "true":
        log(f"[dry-run] would set power limit {watts}W")
        return
    subprocess.run(["nvidia-smi", "-i", cfg["GPU_INDEX"], "-pl", str(watts)],
                   capture_output=True)


def start_miner(cfg):
    if cfg["DRY_RUN"].lower() == "true":
        log(f"[dry-run] would start {cfg['MINER_UNIT']}")
        return
    set_power_limit(cfg["POWER_LIMIT_MINING_W"], cfg)
    subprocess.run(["systemctl", "start", cfg["MINER_UNIT"]], capture_output=True)


def stop_miner(cfg):
    if cfg["DRY_RUN"].lower() == "true":
        log(f"[dry-run] would stop {cfg['MINER_UNIT']}")
        return
    subprocess.run(["systemctl", "stop", cfg["MINER_UNIT"]], capture_output=True)
    # Restore full power immediately so whatever wants the GPU is not throttled.
    set_power_limit(cfg["POWER_LIMIT_IDLE_W"], cfg)


# ------------------------------------------------------------------ main --
def main():
    cfg = load_config()
    windows = parse_windows(cfg["MINE_WINDOWS"])
    poll = float(cfg["POLL_SECONDS"])
    settle_needed = int(cfg["IDLE_SETTLE_POLLS"])
    temp_max = int(cfg["GPU_TEMP_MAX_C"])

    log(f"guard starting | windows={cfg['MINE_WINDOWS']} "
        f"poll={poll}s settle={settle_needed} temp_max={temp_max}C "
        f"unit={cfg['MINER_UNIT']} dry_run={cfg['DRY_RUN']}")

    idle_streak = 0
    last_reason = None
    # Tracked separately from unit_active(): if a start ever fails, we must not
    # retry-and-log on every single poll.
    intended_running = False

    while True:
        reasons = []

        if not in_window(datetime.now(), windows):
            reasons.append(f"outside mining hours ({cfg['MINE_WINDOWS']})")

        if ollama_busy(cfg):
            reasons.append("ollama model loaded (qwen in use)")

        if comfyui_busy(cfg):
            reasons.append("comfyui queue active")

        mining = unit_active(cfg["MINER_UNIT"])
        pids = unit_pids(cfg["MINER_UNIT"]) if mining else set()

        foreign = foreign_gpu_processes(cfg, pids)
        if foreign:
            desc = ", ".join(f"{n} pid {p} ({m} MB)" for p, n, m in foreign[:3])
            reasons.append(f"other GPU work: {desc}")

        t = gpu_temp(cfg)
        if t is not None and t >= temp_max:
            reasons.append(f"gpu {t}C >= {temp_max}C")

        if reasons:
            idle_streak = 0
            if mining or intended_running:
                log(f"STOP: {reasons[0]}")
                stop_miner(cfg)
                intended_running = False
            elif reasons[0] != last_reason:
                log(f"holding: {reasons[0]}")
            last_reason = reasons[0]
        else:
            idle_streak += 1
            last_reason = None
            if not intended_running and idle_streak >= settle_needed:
                log(f"START: idle for {idle_streak} polls, all clear")
                start_miner(cfg)
                intended_running = True
            elif intended_running and not mining and cfg["DRY_RUN"].lower() != "true":
                # Unit died on its own (crash, OOM). Report once, then back off
                # by requiring a fresh settle period before trying again.
                log("miner unit is not running though it should be - will retry")
                intended_running = False
                idle_streak = 0

        time.sleep(poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
