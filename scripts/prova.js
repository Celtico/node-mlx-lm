// Tries the native engine with a real model and measures it.
//   node scripts/prova.js <modelDir> "missatge" [--compare] [--max 200]
// --compare also runs mlx-lm (Python, only for this check) with the same
// prompt ids and greedy sampling, and says whether the tokens are the same.
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { ChatModel } = require('../chat.js');

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const opt = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const [modelDir, message = 'Escriu una funció en JavaScript que sumi els números d\'una llista.'] = args.filter((a, i) => !a.startsWith('--') && !(args[i - 1] ?? '').startsWith('--max'));
if (!modelDir) {
  console.error('Ús: node scripts/prova.js <carpeta del model> "missatge" [--compare] [--max 200]');
  process.exit(1);
}
const maxTokens = Number(opt('--max', 200));

// With --compare, the native run goes in a child process that exits (and frees
// the model) before mlx-lm loads it: two 9B models at once do not fit in 16 GB.
if (flag('--compare') && !process.env.PROVA_DUMP) {
  const dump = path.join(fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'prova-')), 'run.json');
  execFileSync(process.execPath, [__filename, ...args.filter((a) => a !== '--compare')], {
    stdio: 'inherit', env: { ...process.env, PROVA_DUMP: dump },
  });
  const { messages, ids, out } = JSON.parse(fs.readFileSync(dump, 'utf8'));
  fs.rmSync(path.dirname(dump), { recursive: true });
  compare(messages, ids, out);
  process.exit(0);
}

let t0 = Date.now();
const chat = new ChatModel(modelDir);
console.log(`Model carregat en ${((Date.now() - t0) / 1000).toFixed(1)} s (Metal: ${require('@celtico/mlx-node').metal})`);

const messages = [{ role: 'user', content: message }];
const ids = chat.encode(chat.prompt(messages));
const engine = chat.model.engine;

t0 = Date.now();
engine.start(Int32Array.from(ids), {});
const out = [];
let first;
for (let i = 0; i < maxTokens; i++) {
  const id = engine.next();
  if (i === 0) first = Date.now();
  if (chat.model.eos.has(id)) break;
  out.push(id);
}
const end = Date.now();
console.log('\n' + chat.tokenizer.decode(out, { skip_special_tokens: false }) + '\n');
const mem = engine.memory();
console.log(`Prompt: ${ids.length} tokens en ${((first - t0) / 1000).toFixed(2)} s (${(ids.length / ((first - t0) / 1000)).toFixed(0)} tok/s)`);
console.log(`Generació: ${out.length} tokens, ${((out.length - 1) / ((end - first) / 1000)).toFixed(1)} tok/s`);
console.log(`Memòria: activa ${(mem.active / 2 ** 30).toFixed(2)} GB, pic ${(mem.peak / 2 ** 30).toFixed(2)} GB`);

// Free the model now (the process exiting would too) before mlx-lm loads its own.
if (process.env.PROVA_DUMP) chat.model.close();
if (process.env.PROVA_DUMP) fs.writeFileSync(process.env.PROVA_DUMP, JSON.stringify({ messages, ids, out }));

function compare(messages, ids, out) {
  const python = process.env.MLX_PYTHON || 'python3';
  const script = `
import json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
d = json.load(sys.stdin)
model, _ = load(d["dir"])
out = []
for tok, _ in generate_step(mx.array(d["ids"]), model, max_tokens=d["n"], sampler=lambda x: mx.argmax(x, axis=-1)):
    out.append(int(tok))
print(json.dumps(out))
`;
  // Tokenizer + chat template: JS vs transformers (what mlx-lm uses).
  const tokScript = `
import json, sys
from mlx_lm.utils import load_tokenizer
from pathlib import Path
d = json.load(sys.stdin)
tok = load_tokenizer(Path(d["dir"]))
print(json.dumps(tok.apply_chat_template(d["messages"], add_generation_prompt=True, return_dict=False)))
`;
  const pyIds = JSON.parse(execFileSync(python, ['-c', tokScript], { input: JSON.stringify({ dir: path.resolve(modelDir), messages }), encoding: 'utf8' }).trim().split('\n').pop());
  const tokSame = pyIds.length === ids.length && pyIds.every((id, i) => id === ids[i]);
  console.log(tokSame ? `Tokenitzador: idèntic a Python (${ids.length} tokens)` : `Tokenitzador: DIFERENT (JS ${ids.length}, Python ${pyIds.length} tokens)`);
  const n = out.length;
  const ref = JSON.parse(execFileSync(python, ['-c', script], { input: JSON.stringify({ dir: path.resolve(modelDir), ids, n }), encoding: 'utf8', maxBuffer: 1 << 24 }).trim().split('\n').pop());
  const same = ref.slice(0, n).findIndex((id, i) => id !== out[i]);
  console.log(same === -1 ? `mlx-lm: els ${n} tokens són idèntics` : `mlx-lm: difereix al token ${same} (natiu ${out[same]}, mlx-lm ${ref[same]})`);
}
