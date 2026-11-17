// trainLora vs `mlx_lm lora` on the tiny model.
// Make the references: python test/make_tiny.py <dir> && python test/make_lora_ref.py <dir>
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { ChatModel } = require('../chat.js');
const { trainLora } = require('../lora.js');
const { readSafetensors } = require('./safetensors.js');

const dir = process.env.TINY_MODEL;
const skip = (!dir || !fs.existsSync(path.join(dir ?? '', 'reference_lora.json'))) && 'LoRA reference not made';

function maxDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

test('trainLora gives the same adapters as `mlx_lm lora`', { skip }, async () => {
  const refDir = path.join(dir, 'mlx_lm_trained');
  const example = JSON.parse(fs.readFileSync(path.join(refDir, 'example.json'), 'utf8'));
  const chat = new ChatModel(dir);
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'lora-'));
  const reports = [];
  await trainLora(chat, [example], {
    adapterPath: out, iters: 3, learningRate: 1e-3, numLayers: 4, rank: 4, scale: 20,
    maskPrompt: true, resumeAdapterFile: path.join(dir, 'adapters_init.safetensors'),
    stepsPerReport: 1, onReport: (r) => reports.push(r),
  });
  assert.strictEqual(reports.length, 3);
  const got = readSafetensors(path.join(out, 'adapters.safetensors'));
  const diffTo = (ref) => {
    const want = readSafetensors(path.join(ref, 'adapters.safetensors'));
    assert.deepStrictEqual(Object.keys(got).sort(), Object.keys(want).sort());
    let worst = 0;
    for (const k of Object.keys(want)) worst = Math.max(worst, maxDiff(got[k].values, want[k].values));
    return worst;
  };
  // Same ops as `mlx_lm lora` with mx.compile off: identical. With compile,
  // fused kernels round bf16 differently, so only close.
  const exact = diffTo(path.join(dir, 'mlx_lm_trained_nocompile'));
  const compiled = diffTo(refDir);
  console.log(`losses ${reports.map((r) => r.trainLoss.toFixed(4)).join(', ')}; max diff vs mlx_lm lora: ${exact} (no compile), ${compiled} (compiled)`);
  assert.ok(exact < 1e-6, `adapters differ by ${exact} from mlx_lm lora without compile`);
  assert.ok(compiled < 0.02, `adapters differ by ${compiled} from mlx_lm lora`);

  // The result loads back as a beta.
  const cfg = JSON.parse(fs.readFileSync(path.join(out, 'adapter_config.json'), 'utf8'));
  assert.strictEqual(cfg.num_layers, 4);
  chat.model.loadAdapters(out);
  fs.rmSync(out, { recursive: true });
});
