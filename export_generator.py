"""
export_generator.py
===================
One-time converter: turns your trained checkpoint into 'generator.onnx',
a file that runs directly in web browsers (via onnxruntime-web).

Run this on your PC in the same folder as image_ai_trainer.py,
AFTER training (it reads output/checkpoint.pt):

  pip install onnx
  python export_generator.py

Then upload the resulting 'generator.onnx' to your website repo,
in the SAME folder as imageai.html.

Re-run + re-upload whenever you train the model further and want the
website to use the smarter version.
"""

import os

import torch

from image_ai_trainer import Generator, LATENT_DIM, CKPT_PATH

OUT_PATH = "generator.onnx"


def main():
    if not os.path.exists(CKPT_PATH):
        raise SystemExit("No trained model found (output/checkpoint.pt). "
                         "Train first with image_ai_trainer.py.")

    gen = Generator()
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    gen.load_state_dict(ckpt["gen"])
    gen.eval()

    dummy_noise = torch.randn(1, LATENT_DIM, 1, 1)

    torch.onnx.export(
        gen,
        dummy_noise,
        OUT_PATH,
        input_names=["noise"],
        output_names=["image"],
        dynamic_axes={"noise": {0: "batch"}, "image": {0: "batch"}},
        opset_version=13,
        dynamo=False,  # forces ONE self-contained file (no .onnx.data sidecar)
    )

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Exported {OUT_PATH} ({size_mb:.1f} MB) from epoch "
          f"{ckpt.get('epoch', '?')} checkpoint.")
    print("Upload it next to imageai.html in your website repo.")


if __name__ == "__main__":
    main()
