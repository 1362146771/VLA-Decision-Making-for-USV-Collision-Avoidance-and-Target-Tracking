#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, json, argparse, math, numpy as np
from collections import Counter, defaultdict

def parse_args():
    ap = argparse.ArgumentParser("Analyze action_targets distribution")
    ap.add_argument("--data_root", required=True, help="数据根目录，如 /root/autodl-tmp/数据")
    ap.add_argument("--jsonl", required=True, help="samples.jsonl 路径")
    ap.add_argument("--v_act", type=int, default=6, help="动作类别数（直方图长度）")
    ap.add_argument("--by_log", action="store_true", help="是否按 log 段统计")
    ap.add_argument("--log_min", type=int, default=None, help="仅统计这个下界以上的 logN（含）")
    ap.add_argument("--log_max", type=int, default=None, help="仅统计这个上界以下的 logN（含）")
    ap.add_argument("--limit", type=int, default=None, help="最多读取多少条（调试用）")
    ap.add_argument("--save_csv", default=None, help="可选：保存全局+分log统计到 CSV 路径")
    return ap.parse_args()

def get_log_id_from_path(p: str):
    m = re.search(r"/log(\d+)/", p.replace("\\","/"))
    return int(m.group(1)) if m else None

def find_action_targets_path(item, data_root):
    # 兼容多种字段：
    # 1) 直接有 "action_targets_npy"
    # 2) 只有 "actions": actions/log3/frame_00011.npy → 推导成 .targets.npy
    # 3) 直接存在 "action_targets"（少见）
    if "action_targets_npy" in item:
        return os.path.join(data_root, item["action_targets_npy"])
    if "action_targets" in item and isinstance(item["action_targets"], str) and item["action_targets"].endswith(".npy"):
        # 有些数据集直接叫这个名
        return os.path.join(data_root, item["action_targets"])
    if "actions" in item and isinstance(item["actions"], str) and item["actions"].endswith(".npy"):
        p = item["actions"]
        base, _ = os.path.splitext(p)
        guess = base + ".targets.npy"
        abs_guess = os.path.join(data_root, guess)
        if os.path.isfile(abs_guess):
            return abs_guess
    return None

def main():
    args = parse_args()
    total_hist = np.zeros(args.v_act, dtype=np.int64)
    per_log_hist = defaultdict(lambda: np.zeros(args.v_act, dtype=np.int64))
    per_log_counts = Counter()

    seen = 0
    kept = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if args.limit and kept >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            seen += 1
            try:
                it = json.loads(line)
            except Exception:
                continue

            tgt_path = find_action_targets_path(it, args.data_root)
            if not tgt_path or not os.path.isfile(tgt_path):
                continue

            # 过滤 log 范围
            log_id = get_log_id_from_path(tgt_path)
            if args.log_min is not None and (log_id is None or log_id < args.log_min):
                continue
            if args.log_max is not None and (log_id is None or log_id > args.log_max):
                continue

            try:
                arr = np.load(tgt_path)
            except Exception:
                continue

            # arr 形状一般是 [K]，值域 0..v_act-1 或 -100 忽略
            arr = arr.reshape(-1)
            arr = arr[arr >= 0]  # 去掉 ignore_index
            if arr.size == 0:
                continue

            kept += 1
            # 截断到 v_act（防越界）
            arr = arr[(arr < args.v_act)]
            if arr.size == 0:
                continue

            # 累加直方图
            h, _ = np.histogram(arr, bins=np.arange(args.v_act+1))
            total_hist += h
            if args.by_log and log_id is not None:
                per_log_hist[log_id] += h
                per_log_counts[log_id] += arr.size

    # ===== 打印全局 =====
    s = int(total_hist.sum())
    print(f"[global] counted_samples={kept}  tokens={s}")
    if s > 0:
        pct = (total_hist / s) * 100.0
        print("class\tcount\tpercent")
        for c in range(args.v_act):
            print(f"{c}\t{int(total_hist[c])}\t{pct[c]:.2f}%")
        maj = int(total_hist.argmax())
        print(f"[global] majority={maj} ({pct[maj]:.2f}%)  entropy={-(np.where(total_hist>0, total_hist/s, 1e-12)*np.log(np.where(total_hist>0, total_hist/s, 1e-12))).sum():.3f} nats")

    # ===== 分 log =====
    rows = []
    if args.by_log and per_log_hist:
        print("\n[per-log]")
        print("log_id\ttokens\tmaj_cls\tmaj_pct\tcls0..")
        for log_id in sorted(per_log_hist.keys()):
            h = per_log_hist[log_id]
            n = int(h.sum())
            if n == 0:
                continue
            p = (h / n) * 100.0
            maj = int(h.argmax())
            row = [log_id, n, maj, f"{p[maj]:.2f}%"] + [int(x) for x in h.tolist()]
            rows.append(row)
            print(f"{log_id}\t{n}\t{maj}\t{p[maj]:.2f}%\t{h.tolist()}")

    # 可选保存 CSV
    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(["scope","log_id","tokens","maj_cls","maj_pct"] + [f"class{c}" for c in range(args.v_act)])
            # global
            if s > 0:
                cw.writerow(["global","",s,int(total_hist.argmax()), f"{(total_hist.max()/s)*100:.2f}%"] + total_hist.tolist())
            # per-log
            for row in rows:
                log_id, n, maj, majpct, *counts = row
                cw.writerow(["log",log_id,n,maj,majpct] + counts)
        print(f"[wrote] {args.save_csv}")

if __name__ == "__main__":
    main()
