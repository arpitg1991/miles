#!/usr/bin/env python3
"""Rollout-like load for the glm53 memprobe (stdlib only: urllib + threads).

Keeps --concurrency /generate requests in flight for --minutes: each prompt is
a random token-id sequence of [--prompt-min, --prompt-max] tokens passed as
``input_ids`` (GenerateReqInput.input_ids, sglang io_struct.py:171),
max_new_tokens 2048, temperature 1.0, return_logprob true, top_logprobs_num 0,
stream false -- the shape of the arena rollout (use_rollout_logprobs, 32k-per-turn
re-prefills, radix cache off). A fraction (--abort-frac) of requests is aborted
client-side after a random 15-60 s via POST /abort_request {"rid": ...}
(http_server.py:1694; AbortReq io_struct.py:2064) to mimic the run's
partial-rollout/client-disconnect aborts ("state was deleted in TokenizerManager").

Prints a one-line status every 60 s and appends one CSV row per request.
"""
import argparse
import csv
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

# GLM-5.3-Flash vocab 154880; eos ids 154820/154827/154829 are outside this range
VOCAB_LO, VOCAB_HI = 1000, 150000


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.completed = 0
        self.aborted = 0
        self.errors = 0
        self.inflight = 0
        self.gen_tokens = 0
        self.prompt_tokens = 0
        self.lat_sum = 0.0
        self.started = 0


def post_json(url, obj, timeout):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def worker(idx, args, stats, stop_at, writer, wfile, wlock, log):
    rng = random.Random(idx * 7919 + int(time.time()))
    while time.time() < stop_at:
        plen = rng.randint(args.prompt_min, args.prompt_max)
        ids = [rng.randint(VOCAB_LO, VOCAB_HI) for _ in range(plen)]
        rid = "memprobe-%d-%s" % (idx, uuid.uuid4().hex[:12])
        body = {
            "rid": rid,
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": args.max_new_tokens, "temperature": 1.0, "ignore_eos": args.ignore_eos},
            "return_logprob": True,
            "top_logprobs_num": 0,
            "stream": False,
        }
        will_abort = rng.random() < args.abort_frac
        abort_timer = None
        aborted = {"v": False}
        if will_abort:
            delay = rng.uniform(15, 60)

            def do_abort(rid=rid):
                try:
                    post_json(args.base_url + "/abort_request", {"rid": rid}, 30)
                    aborted["v"] = True
                except Exception as e:  # noqa: BLE001
                    log("abort %s failed: %s" % (rid, e))

            abort_timer = threading.Timer(delay, do_abort)
            abort_timer.daemon = True
            abort_timer.start()
        t0 = time.time()
        with stats.lock:
            stats.inflight += 1
            stats.started += 1
        status = "ok"
        gen = 0
        err = ""
        try:
            code, raw = post_json(args.base_url + "/generate", body, args.timeout)
            try:
                out = json.loads(raw)
            except ValueError:
                out = {}
            if isinstance(out, dict) and "error" in out:
                status = "error"
                err = str(out["error"])[:200]
            else:
                mi = out.get("meta_info", {}) if isinstance(out, dict) else {}
                gen = int(mi.get("completion_tokens", 0) or 0)
                fr = mi.get("finish_reason") or {}
                if isinstance(fr, dict) and fr.get("type") == "abort":
                    status = "aborted"
        except urllib.error.HTTPError as e:
            body_txt = e.read()[:300].decode(errors="replace")
            if aborted["v"] or "abort" in body_txt.lower():
                status = "aborted"
            else:
                status = "http%d" % e.code
                err = body_txt[:200]
        except Exception as e:  # noqa: BLE001
            status = "aborted" if aborted["v"] else "exc"
            err = repr(e)[:200]
        finally:
            if abort_timer is not None:
                abort_timer.cancel()
        lat = time.time() - t0
        if aborted["v"] and status == "ok":
            status = "aborted"
        with stats.lock:
            stats.inflight -= 1
            stats.prompt_tokens += plen
            stats.gen_tokens += gen
            stats.lat_sum += lat
            if status == "ok":
                stats.completed += 1
            elif status == "aborted":
                stats.aborted += 1
            else:
                stats.errors += 1
        with wlock:
            writer.writerow([now(), rid, plen, gen, status, "%.1f" % lat, err])
            wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--prompt-min", type=int, default=40000)
    ap.add_argument("--prompt-max", type=int, default=100000)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--abort-frac", type=float, default=0.10)
    ap.add_argument("--ignore-eos", type=int, default=1, help="1: decode the full max_new_tokens (random prompts would otherwise stop early)")
    ap.add_argument("--timeout", type=float, default=7200, help="per-request HTTP timeout (queues are deep at 96 x 100k tokens)")
    ap.add_argument("--csv", default="loadgen.csv")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    args.ignore_eos = bool(args.ignore_eos)

    logf = open(args.log, "a") if args.log else None

    def log(msg):
        line = "%s %s" % (now(), msg)
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    f = open(args.csv, "a", newline="")
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["ts_end", "rid", "prompt_tokens", "gen_tokens", "status", "latency_s", "error"])
    wlock = threading.Lock()
    stats = Stats()
    stop_at = time.time() + args.minutes * 60
    log("loadgen start: concurrency=%d prompt=[%d,%d] max_new_tokens=%d abort_frac=%.2f minutes=%.0f url=%s"
        % (args.concurrency, args.prompt_min, args.prompt_max, args.max_new_tokens, args.abort_frac, args.minutes, args.base_url))
    threads = []
    for i in range(args.concurrency):
        t = threading.Thread(target=worker, args=(i, args, stats, stop_at, writer, f, wlock, log), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.05)  # stagger the initial burst slightly
    t_last = time.time()
    g_last = stats.gen_tokens
    t_report = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        if time.time() - t_report < 60 and any(t.is_alive() for t in threads):
            continue
        t_report = time.time()
        with stats.lock:
            g = stats.gen_tokens
            done = stats.completed + stats.aborted + stats.errors
            avg = stats.lat_sum / done if done else 0.0
            msg = ("status: started=%d inflight=%d completed=%d aborted=%d errors=%d gen_tokens=%d (%.0f tok/s last min) "
                   "prompt_tokens=%d avg_latency=%.0fs remaining=%.0fmin"
                   % (stats.started, stats.inflight, stats.completed, stats.aborted, stats.errors, g,
                      (g - g_last) / max(1.0, time.time() - t_last), stats.prompt_tokens, avg, max(0.0, (stop_at - time.time()) / 60)))
        log(msg)
        t_last = time.time(); g_last = g
        f.flush()
        # the deadline has passed and only stragglers remain: give them a bounded grace period
        if time.time() > stop_at + args.timeout:
            log("giving up on stragglers")
            break
    with stats.lock:
        log("loadgen end: completed=%d aborted=%d errors=%d gen_tokens=%d prompt_tokens=%d"
            % (stats.completed, stats.aborted, stats.errors, stats.gen_tokens, stats.prompt_tokens))
    f.close()
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
