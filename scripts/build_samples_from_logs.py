#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将按轨迹分文件的数据构造成 OceanVLA 训练清单（仅处理指定区间的 log1~logN）：
  root/
    samples.jsonl
    actions/log{ID}/frame_xxxxx.npy      # [K,7]
    wm_targets/log{ID}/frame_xxxxx.npy   # [K*M]（可选随机占位）
要求：
- 每个轨迹有两份文件：
  描述:   描述/descriptions_openvla_log{ID}.jsonl
  数据对: 数据对/openvla_log{ID}.json 或 .jsonl
- 图片通常在 IMAGES/log{ID}/frame_xxx.png（若无则退回 IMAGES/frame_xxx.png）
"""

import os, re, json, argparse
import numpy as np

def load_json_any(path: str):
    """兼容 .json / .jsonl，统一返回 list[dict]."""
    items = []
    if not os.path.isfile(path):
        return items
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        items.append(json.loads(ln))
                    except Exception:
                        pass
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "root" in data:
            items = data["root"]
        elif isinstance(data, list):
            items = data
        else:
            items = [data]
    return items

def build_desc_index_for_log(desc_path: str):
    """
    读 descriptions_openvla_log{ID}.jsonl
    返回 {basename(image): description}
    """
    desc_map = {}
    items = load_json_any(desc_path)
    for it in items:
        img = it.get("image") or it.get("frame") or ""
        if not img:
            continue
        key = os.path.basename(img)
        desc = (it.get("description") or it.get("text") or "").strip()
        if desc:
            desc_map[key] = desc
    return desc_map

def group_items_in_log(pairs_items, K: int):
    """
    在同一个 log 的数据对内做分组：
    - 优先用 meta.group_index / frame_index 排序，每 K 条合成一组（不重叠）
    - 若没有 meta，按出现顺序每 K 条一组（不重叠）
    返回：list[list[dict]]，每组长度==K
    """
    has_meta = any(("meta" in it) and it["meta"] and ("group_index" in it["meta"]) for it in pairs_items)
    groups, buf = [], []

    if has_meta:
        items = sorted(
            [it for it in pairs_items if it.get("meta")],
            key=lambda x: (x["meta"].get("group_index", 0), x["meta"].get("frame_index", 0))
        )
    else:
        items = pairs_items

    for it in items:
        buf.append(it)
        if len(buf) == K:
            groups.append(buf)
            buf = []
    return groups

def vec_from_action(a: dict, a_dim: int):
    """把 action 字段转成长度 a_dim 的向量（前7维是既定顺序，不足补零）"""
    base = [
        float(a.get("speed_mps_start", 0.0)),
        float(a.get("speed_mps_end", 0.0)),
        float(a.get("delta_speed", 0.0)),
        float(a.get("course_deg_start", 0.0)),
        float(a.get("course_deg_end", 0.0)),
        float(a.get("delta_course_deg", 0.0)),
        float(a.get("avg_speed_mps", 0.0)),
    ]
    if len(base) < a_dim:
        base += [0.0] * (a_dim - len(base))
    return np.asarray(base[:a_dim], dtype=np.float32)

def synthesize_desc(group):
    """兜底：若这一组没有任何自然语言描述，就根据动作合成一句中文。"""
    parts = []
    for it in group:
        a = it.get("action", {}) or {}
        t = a.get("type", "前进")
        v0 = a.get("speed_mps_start", 0.0)
        v1 = a.get("speed_mps_end", 0.0)
        c0 = a.get("course_deg_start", 0.0)
        c1 = a.get("course_deg_end", 0.0)
        parts.append(f"{t}，速度 {v0:.1f}->{v1:.1f} m/s，航向 {c0:.1f}->{c1:.1f}°")
    return " 然后 ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据根目录，例如 /root/autodl-tmp/数据")
    ap.add_argument("--images_dir", default="IMAGES", help="图像相对目录")
    ap.add_argument("--pairs_dir", default="数据对", help="数据对相对目录（openvla_log{ID}.json[l]）")
    ap.add_argument("--desc_dir", default="描述", help="描述相对目录（descriptions_openvla_log{ID}.jsonl）")
    ap.add_argument("--out_jsonl", default="samples.jsonl", help="输出清单文件名")
    ap.add_argument("--k", type=int, default=4, help="每组帧数（不重叠分组）")
    ap.add_argument("--a_dim", type=int, default=7, help="动作维度")
    ap.add_argument("--m_per_frame", type=int, default=256, help="VQ 每帧 token 数")
    ap.add_argument("--v_world", type=int, default=16384, help="VQ 码表大小")
    ap.add_argument("--write_random_wm", action="store_true", help="写入随机 wm_targets .npy")
    ap.add_argument("--min_log_id", type=int, default=1)
    ap.add_argument("--max_log_id", type=int, default=500)
    args = ap.parse_args()

    root = args.root
    P_IMAGES = os.path.join(root, args.images_dir)
    P_PAIRS  = os.path.join(root, args.pairs_dir)
    P_DESC   = os.path.join(root, args.desc_dir)
    OUT_JSONL = os.path.join(root, args.out_jsonl)

    actions_root = os.path.join(root, "actions")
    wm_root = os.path.join(root, "wm_targets")
    os.makedirs(actions_root, exist_ok=True)
    if args.write_random_wm:
        os.makedirs(wm_root, exist_ok=True)

    H_world = args.k * args.m_per_frame
    n_written_total = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as fout:
        for lid in range(args.min_log_id, args.max_log_id + 1):
            # 文件路径（按 logID 精确匹配）
            pairs_path = None
            cand_pairs = [
                os.path.join(P_PAIRS, f"openvla_log{lid}.jsonl"),
                os.path.join(P_PAIRS, f"openvla_log{lid}.json"),
            ]
            for cp in cand_pairs:
                if os.path.isfile(cp):
                    pairs_path = cp
                    break
            if pairs_path is None:
                print(f"[WARN] 缺少数据对文件: openvla_log{lid}.json[l]，跳过该 log")
                continue

            desc_path = os.path.join(P_DESC, f"descriptions_openvla_log{lid}.jsonl")
            desc_map = build_desc_index_for_log(desc_path)

            pairs_items = load_json_any(pairs_path)
            if not pairs_items:
                print(f"[WARN] 数据对为空: {pairs_path}，跳过该 log")
                continue

            groups = group_items_in_log(pairs_items, K=args.k)
            if not groups:
                print(f"[WARN] log{lid} 未形成任何分组（可能样本不足 {args.k}），跳过该 log")
                continue

            actions_dir = os.path.join(actions_root, f"log{lid}")
            os.makedirs(actions_dir, exist_ok=True)
            if args.write_random_wm:
                wm_dir = os.path.join(wm_root, f"log{lid}")
                os.makedirs(wm_dir, exist_ok=True)

            n_written_log = 0

            for g in groups:
                img0 = g[0].get("image") or ""
                if not img0:
                    continue
                stem = os.path.splitext(os.path.basename(img0))[0]

                # 文本：优先取描述（按文件名 basename 对齐），没有则合成
                texts = []
                for it in g:
                    key = os.path.basename(it.get("image","") or "")
                    t = desc_map.get(key, "")
                    if t:
                        texts.append(t)
                text = " 然后 ".join(texts) if texts else synthesize_desc(g)

                # 动作矩阵 [K, a_dim]
                acts = np.zeros((args.k, args.a_dim), dtype=np.float32)
                for i, it in enumerate(g[:args.k]):
                    a = it.get("action", {}) or {}
                    acts[i] = vec_from_action(a, args.a_dim)

                # 写动作
                rel_actions = f"actions/log{lid}/{stem}.npy"
                np.save(os.path.join(root, rel_actions), acts)

                # 写 wm_targets（可选随机占位）
                if args.write_random_wm:
                    wm = np.random.randint(0, args.v_world, size=(H_world,), dtype=np.int64)
                    rel_wm = f"wm_targets/log{lid}/{stem}.npy"
                    np.save(os.path.join(root, rel_wm), wm)
                else:
                    rel_wm = None

                # 图像相对路径：优先 IMAGES/log{lid}/frame_xxx.png，否则回退 IMAGES/frame_xxx.png
                candidate1 = os.path.join(args.images_dir, f"log{lid}", os.path.basename(img0))
                candidate2 = os.path.join(args.images_dir, os.path.basename(img0))
                abs1 = os.path.join(root, candidate1)
                rel_image = candidate1 if os.path.isfile(abs1) else candidate2

                # 写 1 条样本
                sample = {
                    "image": rel_image,
                    "text": text,
                    "actions": rel_actions,
                    "wm_targets": rel_wm
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_written_log += 1
                n_written_total += 1

            print(f"[OK] log{lid}: 写出 {n_written_log} 条")

    print(f"[DONE] 总计写出 {n_written_total} 条 -> {OUT_JSONL}")
    if not args.write_random_wm:
        print("[提示] 未写 wm_targets；训练时需在 Dataset 中开启随机占位或换成真实 VQ tokens")

if __name__ == "__main__":
    main()