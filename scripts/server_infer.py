# /root/server_infer.py
# -*- coding: utf-8 -*-
import os, socket, struct, json, tempfile, subprocess, sys, time, shutil

HOST = "127.0.0.1"   # 只监听回环
INFER_PORT = 6006    # 接收图片
RETURN_HOST = "127.0.0.1"  # 通过 -R 转回本地
RETURN_PORT = 6007

# === 你环境里的固定参数（请按需改成你的路径/超参） ===
PYTHON = sys.executable
MODULE = "core.scripts.eval_min"
CKPT = "checkpoints/your_ckpt/oceanvla_step10000.pt"
TOKENIZER_PATH = "/root/autodl-tmp/models--google--gemma-2b"
CLIP_PATH = "/root/autodl-tmp/models--openai--clip-vit-base-patch32"
DATA_ROOT = "/root/autodl-tmp"  # 只要 JSONL 里的 image 用绝对路径，DATA_ROOT 也可设为 "/"

# === 约定的传输协议：先发 8 字节文件名长度，再发 8 字节数据长度，然后依次发文件名(UTF-8)与数据 ===
HEADER_FMT = "!QQ"  # 两个 uint64 大端

def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf

def run_eval_on(image_abspath):
    # 为单张图生成一个临时 JSONL
    workdir = tempfile.mkdtemp(prefix="live_eval_")
    try:
        jsonl = os.path.join(workdir, "live.jsonl")
        record = {"image": image_abspath, "text": "", "meta": {"log_id": "live", "group_index": 0, "group_size": 1}}
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        cmd = [
            PYTHON, "-m", MODULE,
            "--data_root", DATA_ROOT,
            "--jsonl", jsonl,
            "--ckpt", CKPT,
            "--batch_size", "1", "--num_batches", "1",
            "--k_act", "1", "--a_dim", "6", "--v_act", "6",
            "--h_world", "32", "--v_wm", "8192",
            "--use_clip_processor", "--clip_path", CLIP_PATH,
            "--temporal_enabled", "--num_frames", "4", "--stride", "20",
            "--actions_key", "__disable_actions__",
            "--wm_tgt_hw", "8,4", "--wm_pool", "mode",
            "--apply_la_infer", "--la_tau_infer", "1.0",
            "--amp", "--amp_dtype", "bf16",
            "--emit_predictions"
        ]
        env = os.environ.copy()
        env["TOKENIZER_PATH"] = TOKENIZER_PATH  # 你的 eval_min 已有默认值；这里留作备用

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        top1 = None
        raw_lines = []
        for line in p.stdout:
            raw_lines.append(line.rstrip("\n"))
            if line.startswith("JSON_RESULT "):
                try:
                    payload = json.loads(line[len("JSON_RESULT "):])
                    top1 = int(payload.get("top1"))
                except Exception:
                    pass
        p.wait()

        # 组装一个回传结果（若没捕到 JSON，就把stdout合并回传）
        if top1 is not None:
            return {"ok": True, "pred_action": top1, "image": image_abspath}
        else:
            return {"ok": False, "stderr": "\n".join(raw_lines), "image": image_abspath}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def send_back(result_dict):
    # 通过反向转发口把 JSON 结果回到本地
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect((RETURN_HOST, RETURN_PORT))
    payload = json.dumps(result_dict, ensure_ascii=False).encode("utf-8")
    s.sendall(struct.pack("!Q", len(payload)))
    s.sendall(payload)
    s.close()

def main():
    os.makedirs("/root/autodl-tmp/inbox", exist_ok=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, INFER_PORT))
    srv.listen(16)
    print(f"[server] listening on {HOST}:{INFER_PORT}")
    while True:
        conn, addr = srv.accept()
        try:
            # 收头
            hdr = recv_exact(conn, struct.calcsize(HEADER_FMT))
            name_len, data_len = struct.unpack(HEADER_FMT, hdr)
            # 收文件名
            fname = recv_exact(conn, name_len).decode("utf-8", errors="ignore")
            # 收数据
            data = recv_exact(conn, data_len)
            # 落盘
            save_path = os.path.join("/root/autodl-tmp/inbox", os.path.basename(fname))
            with open(save_path, "wb") as f:
                f.write(data)
            print(f"[server] received image -> {save_path} ({len(data)} bytes)")
            # 推理
            result = run_eval_on(os.path.abspath(save_path))
            # 回传
            send_back(result)
            # 给发送端一个简单 ACK
            conn.sendall(b"OK")
        except Exception as e:
            try:
                send_back({"ok": False, "error": repr(e)})
            except Exception:
                pass
        finally:
            conn.close()

if __name__ == "__main__":
    main()
