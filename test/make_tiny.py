# Builds a tiny random Qwen3.5 model in mlx-community layout (mixed 4/8-bit),
# runs it with mlx-lm and saves reference logits for the native engine test.
import json, sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import save_model, save_config

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
mx.random.seed(0)

text = dict(
    model_type="qwen3_5_text", hidden_size=256, intermediate_size=512,
    num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, rms_norm_eps=1e-6, vocab_size=512,
    linear_num_value_heads=4, linear_num_key_heads=2,
    linear_key_head_dim=128, linear_value_head_dim=64,
    linear_conv_kernel_dim=4, full_attention_interval=4,
    tie_word_embeddings=False,
    rope_parameters={"type": "default", "mrope_section": [11, 11, 10],
                     "mrope_interleaved": True, "rope_theta": 10000000,
                     "partial_rotary_factor": 0.25},
)
config = {"model_type": "qwen3_5", "text_config": text}
model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
# Random non-trivial values for the parameters that start as ones/zeros.
for i, layer in enumerate(model.language_model.model.layers):
    if layer.is_linear:
        la = layer.linear_attn
        la.dt_bias = mx.random.normal(la.dt_bias.shape) * 0.5
        la.norm.weight = 1 + 0.1 * mx.random.normal(la.norm.weight.shape)
    layer.input_layernorm.weight = 1 + 0.1 * mx.random.normal((256,))
model.set_dtype(mx.bfloat16)
for layer in model.language_model.model.layers:
    if layer.is_linear:
        layer.linear_attn.A_log = layer.linear_attn.A_log.astype(mx.float32)

def predicate(path, m):
    if not hasattr(m, "to_quantized"):
        return False
    eight = ("embed_tokens" in path or "lm_head" in path or "in_proj_b" in path
             or "in_proj_a" in path or "out_proj" in path or "v_proj" in path or ".0.mlp" in path)
    return {"group_size": 64, "bits": 8 if eight else 4}

qcfg = {}
def record(path, m):
    r = predicate(path, m)
    if r:
        qcfg[path] = r
    return r
nn.quantize(model, class_predicate=record)
config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine", **qcfg}
config["eos_token_id"] = [1]

save_model(out, model)
save_config(config, out / "config.json")

tokens = mx.array([[5, 17, 300, 42, 7, 99, 128, 3, 250, 11, 64, 400]])
cache = make_prompt_cache(model)
logits = model(tokens[:, :8], cache=cache)
steps = [logits[0, -1]]
for t in range(8, tokens.shape[1]):
    l = model(tokens[:, t:t+1], cache=cache)
    steps.append(l[0, -1])
ref = mx.stack(steps).astype(mx.float32)
mx.eval(ref)
json.dump({"tokens": tokens[0].tolist(), "prefill": 8,
           "logits": ref.tolist()}, open(out / "reference.json", "w"))
print("saved", out, ref.shape, "argmax", mx.argmax(ref, -1).tolist())

# Long prompt: exercises chunked prefill (prefill_step_size 256) and KV cache growth.
from mlx_lm.generate import generate_step
long_prompt = mx.random.randint(2, 512, (700,)).tolist()
greedy = []
for tok, _ in generate_step(mx.array(long_prompt), model, max_tokens=12, prefill_step_size=256,
                            sampler=lambda x: mx.argmax(x, axis=-1)):
    greedy.append(int(tok))
json.dump({"prompt": long_prompt, "greedy": greedy}, open(out / "reference_long.json", "w"))
print("long greedy", greedy)

# Tiny byte-level BPE tokenizer + chat template. It is re-saved through
# transformers so tokenizer.json is exactly what mlx-lm uses (transformers
# applies Qwen3.5's own pre-tokenizer for this model type).
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import AutoTokenizer
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tr = trainers.BpeTrainer(vocab_size=500, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train_from_iterator(["Hola, com estàs? Fes una app amb Chakra UI.\n" * 50,
                         "function main() { return 42; }\n" * 50], tr)
tok.save(str(out / "tokenizer.json"))
template = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
json.dump({"eos_token": "<|im_end|>", "chat_template": template}, open(out / "tokenizer_config.json", "w"))
hf = AutoTokenizer.from_pretrained(str(out))
hf.backend_tokenizer.save(str(out / "tokenizer.json"))
hf = AutoTokenizer.from_pretrained(str(out))
texts = ["Hola, com estàs? 🙂", "function main() { return 42; }", "<|im_start|>user\nFes una app<|im_end|>\n"]
chat = [{"role": "user", "content": "Fes una app"}]
json.dump({
    "texts": [{"text": t, "ids": hf.encode(t, add_special_tokens=False)} for t in texts],
    "chat": chat,
    "prompt": hf.apply_chat_template(chat, add_generation_prompt=True, tokenize=False),
    "prompt_ids": hf.apply_chat_template(chat, add_generation_prompt=True, return_dict=False),
}, open(out / "reference_tokenizer.json", "w"))
print("tokenizer saved")
