"""Bước 4: sinh hình doodle whiteboard bằng mflux (chạy local trên GPU Mac).

Model: z-image-turbo (Tongyi, Apache-2.0) hoặc flux2-klein-4b (Black Forest Labs, Apache-2.0).

Mỗi mô tả sinh vài seed, chấm điểm độ sạch nét (quality.py), giữ ảnh tốt nhất.
Model được nạp 1 lần cho cả loạt ảnh.

Chạy: python -m pipeline.image_gen "a house with a tree" "a sun" --out jobs/gen --seeds 3 [--model flux2-klein-4b]
Đầu ra: <out>/<slug>/seed_<n>.png, best.png, scores.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2

from pipeline import quality

STYLES = {
    # chỉ nét đen — tách nét sạch nhất
    "ink": ("Hand-drawn whiteboard doodle of {subject}. Thick uniform black marker outlines, "
            "simple flat shapes, pure white background, minimal details, no shading, no gradients, "
            "no hatching, no text, no letters, large subject filling most of the frame."),
    # nét đen + tô màu phẳng như bút dạ
    "color": ("Hand-drawn whiteboard doodle of {subject}. Thick uniform black marker outlines filled "
              "with flat solid bright colors (no crayon or pencil texture), simple shapes, pure white background, "
              "minimal details, no shading, no gradients, no hatching, no text, no letters, "
              "large subject filling most of the frame."),
}
# tỉ lệ khớp ô đặt hình trong khung dọc (0.90W x 0.62H của 1080x1920)
WIDTH, HEIGHT = 832, 1024


def slugify(text, n=40):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "image"


MODELS = ["z-image-turbo", "flux2-klein-4b"]


def _parse_cli_args(build_parser, argv):
    """Parser của mflux chỉ đọc sys.argv → tạm thay để dùng lại đúng giá trị mặc định của CLI."""
    saved, sys.argv = sys.argv, ["mflux", *argv]
    try:
        return build_parser().parse_args()
    finally:
        sys.argv = saved


class Generator:
    def __init__(self, model="flux2-klein-4b", quantize=4, steps=None, cache_limit_gb=2):
        import mlx.core as mx

        # MLX giữ lại buffer đã dùng xong để tái sử dụng: không giới hạn thì phình tới ~17GB sau mỗi ảnh
        # (đo với Klein: footprint 20GB → 6.5GB khi giới hạn 2GB, tốc độ không đổi)
        mx.set_cache_limit(int(cache_limit_gb * 1024**3))
        self.name = model
        argv = ["--prompt", "x", "-q", str(quantize), "--width", str(WIDTH), "--height", str(HEIGHT)]
        if model == "z-image-turbo":
            from mflux.models.z_image.cli.z_image_turbo_generate import ZImageTurboCommand, build_parser

            self.args = _parse_cli_args(build_parser, argv + (["--steps", str(steps)] if steps else []))
            self.cmd = ZImageTurboCommand
            self.model = ZImageTurboCommand.load(self.args)
        elif model == "flux2-klein-4b":
            import mflux.models.flux2.cli.flux2_generate as f2

            # Klein là model distilled: 4 bước, guidance cố định 1.0
            self.args = _parse_cli_args(f2.build_parser, argv + ["--steps", str(steps or 4)])
            cfg = f2.ConfigResolution.resolve_restricted(
                model, f2.DEFAULT_MODEL, model_path=None, extra_keys=f2.FAMILY_MODELS, base_model=None)
            self.model = f2.Flux2Klein(model_config=cfg, quantize=quantize, model_path=None,
                                       compute_precision=f2.ComputePrecision.dtype_for(self.args.compute_precision))
        else:
            raise ValueError(f"model không hỗ trợ: {model}")

    def generate(self, prompt, seed, path):
        if self.name == "z-image-turbo":
            image = self.cmd.generate(self.model, self.args, seed, prompt)
        else:
            a = self.args
            image = self.model.generate_image(
                seed=seed, prompt=prompt, width=WIDTH, height=HEIGHT, guidance=1.0, image_path=None,
                num_inference_steps=a.steps, image_strength=a.image_strength,
                scheduler="flow_match_euler_discrete", pid_decode=a.pid_decode, pid_degrade_sigma=a.pid_degrade_sigma)
        image.save(path=str(path))


def generate_best(gen, subject, out_dir, style="color", seeds=3, base_seed=1):
    """Sinh `seeds` ảnh cho 1 chủ thể, chấm điểm, chép ảnh điểm cao nhất thành best.png.
    Chạy lại trên cùng thư mục: seed nối tiếp lần trước, giữ kết quả cũ nếu prompt (chủ thể + style) không đổi."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = STYLES[style].format(subject=subject)
    scores = out_dir / "scores.json"
    prev = json.loads(scores.read_text()) if scores.exists() else {"results": []}
    base_seed = max((r["seed"] for r in prev["results"]), default=base_seed - 1) + 1
    results = prev["results"] if prev.get("prompt") == prompt else []
    for seed in range(base_seed, base_seed + seeds):
        path = out_dir / f"seed_{seed}.png"
        t0 = time.time()
        gen.generate(prompt, seed, path)
        m = quality.assess(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
        m.update(seed=seed, path=path.name, gen_seconds=round(time.time() - t0, 1))
        results.append(m)
        print(f"    seed {seed}: điểm {m['score']:5.1f}  ({m['gen_seconds']}s, {m['n_strokes']} nét)")
    best = max(results, key=lambda r: r["score"])
    (out_dir / "best.png").write_bytes((out_dir / best["path"]).read_bytes())
    scores.write_text(json.dumps({"subject": subject, "prompt": prompt, "results": results}, indent=1))
    return out_dir / "best.png", best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="+", help="mô tả hình (tiếng Anh cho model hiểu tốt nhất)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", choices=list(STYLES), default="color")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, help="số bước khử nhiễu (mặc định của model)")
    ap.add_argument("-q", "--quantize", type=int, default=4)
    ap.add_argument("--model", choices=MODELS, default="flux2-klein-4b")
    args = ap.parse_args()

    t0 = time.time()
    gen = Generator(args.model, args.quantize, args.steps)
    print(f"Nạp model: {time.time() - t0:.0f}s")
    for subject in args.subjects:
        print(f"• {subject}")
        path, best = generate_best(gen, subject, Path(args.out) / slugify(subject), args.style, args.seeds)
        print(f"  → {path} (seed {best['seed']}, điểm {best['score']})")


if __name__ == "__main__":
    main()
