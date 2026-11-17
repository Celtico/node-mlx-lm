# LoRA reference from mlx-lm on the tiny model made by make_tiny.py:
# adapters (with non-zero b), logits with adapters, and 3 training steps
# (mlx_lm default_loss + optim.Adam), saved for the native engine test.
import json, sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import default_loss

d = Path(sys.argv[1])
mx.random.seed(1)
model, _ = load_model(d)
model.freeze()
lora = {"rank": 4, "scale": 20.0, "dropout": 0.0}
linear_to_lora_layers(model, 4, lora)
# Non-zero b so the adapters change the output.
params = dict(tree_flatten(model.trainable_parameters()))
for k in params:
    if k.endswith("lora_b"):
        params[k] = 0.01 * mx.random.normal(params[k].shape)
model.load_weights(list(params.items()), strict=False)
mx.save_safetensors(str(d / "adapters_init.safetensors"), dict(tree_flatten(model.trainable_parameters())))
json.dump({"fine_tune_type": "lora", "num_layers": 4, "lora_parameters": lora}, open(d / "adapter_config.json", "w"))

model.eval()
tokens = mx.array([[5, 17, 300, 42, 7, 99, 128, 3, 250, 11]])
logits = model(tokens)[0, -1].astype(mx.float32)

# Training: batch size 1, padded like iterate_batches (1 + multiple of 32).
model.train()
seqs = [([5, 17, 300, 42, 7, 99, 128, 3, 250, 11, 64, 400, 1], 5),
        ([9, 8, 7, 6, 5, 4, 3, 2, 300, 301, 302, 1], 8),
        ([100, 200, 300, 400, 50, 60, 70, 80, 90, 1], 3)]
opt = optim.Adam(learning_rate=1e-3)
loss_and_grad = nn.value_and_grad(model, default_loss)
steps = []
for toks, offset in seqs:
    L = len(toks)
    padded = 1 + 32 * ((L + 31) // 32)
    arr = toks + [0] * (padded - L)
    batch = mx.array([arr])
    lengths = mx.array([[offset, L]])
    (lv, nt), grad = loss_and_grad(model, batch, lengths)
    opt.update(model, grad)
    mx.eval(model.parameters(), opt.state, lv, nt)
    steps.append({"tokens": arr, "offset": offset, "length": L, "loss": lv.item(), "ntoks": nt.item()})
mx.save_safetensors(str(d / "adapters_final.safetensors"), dict(tree_flatten(model.trainable_parameters())))
json.dump({"tokens": tokens[0].tolist(), "logits": logits.tolist(), "lr": 1e-3, "steps": steps},
          open(d / "reference_lora.json", "w"))
print("losses", [s["loss"] for s in steps])

# End to end: `mlx_lm lora` itself (chat template, mask_prompt, EOS, padding,
# Adam) on one example, resuming from the same initial adapters.
import subprocess, tempfile
work = Path(tempfile.mkdtemp())
(work / "data").mkdir()
example = {"messages": [{"role": "user", "content": "Fes una app"},
                        {"role": "assistant", "content": "function main() { return 42; }"}]}
for f in ["train", "valid"]:
    (work / "data" / f"{f}.jsonl").write_text(json.dumps(example) + "\n")
(work / "lora.yaml").write_text("lora_parameters:\n  rank: 4\n  scale: 20.0\n  dropout: 0.0\n")
# `mlx_lm lora` compiles its train step (mx.compile), which fuses kernels and
# rounds bf16 differently; the native engine does not compile. So there are two
# references: as is, and with mx.compile turned off (the same math, op by op).
NO_COMPILE = ("import sys, runpy, mlx.core as mx; "
              "mx.compile = lambda f=None, **kw: f if f is not None else (lambda g: g); "
              "sys.argv = ['mlx_lm'] + sys.argv[1:]; runpy.run_module('mlx_lm', run_name='__main__')")
for out, launcher in [("mlx_lm_trained", ["-m", "mlx_lm"]), ("mlx_lm_trained_nocompile", ["-c", NO_COMPILE])]:
    subprocess.run([sys.executable, *launcher, "lora", "--model", str(d), "--train",
                    "--data", str(work / "data"), "--iters", "3", "--batch-size", "1",
                    "--num-layers", "4", "--learning-rate", "1e-3", "--mask-prompt",
                    "--resume-adapter-file", str(d / "adapters_init.safetensors"),
                    "--adapter-path", str(d / out), "-c", str(work / "lora.yaml"),
                    "--val-batches", "1", "--steps-per-eval", "1000"], check=True,
                   stdout=subprocess.DEVNULL)
    json.dump(example, open(d / out / "example.json", "w"))
    print("mlx_lm lora reference in", d / out)
