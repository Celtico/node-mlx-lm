// LoRA training from Node, without Python. Same data handling, loss and
// optimizer as `mlx_lm lora` (ChatDataset, default_loss, Adam), and the
// adapters it writes (adapters.safetensors + adapter_config.json) load in
// mlx-lm and in the native engine alike.
const fs = require('node:fs');
const path = require('node:path');

// mlx_lm ChatDataset.process: tokens of the whole chat and, with maskPrompt,
// the offset where the last message starts. (ChatDataset adds no extra EOS.)
function tokenizeExample(chat, example, maskPrompt) {
  const { messages, tools } = example;
  const tokens = chat.encode(chat.template.render({ messages, tools, add_generation_prompt: false, ...chat.specials }));
  let offset = 0;
  if (maskPrompt) {
    const addGen = messages[messages.length - 1]?.role === 'assistant';
    offset = chat.encode(chat.template.render({ messages: messages.slice(0, -1), tools, add_generation_prompt: addGen, ...chat.specials })).length;
  }
  return { tokens, offset };
}

// mlx_lm iterate_batches with batch size 1: truncate and pad to 1 + a
// multiple of 32 (padding is masked out of the loss).
function padExample({ tokens, offset }, maxSeqLength) {
  const length = Math.min(tokens.length, maxSeqLength);
  const padded = Math.min(1 + 32 * Math.ceil(tokens.length / 32), maxSeqLength);
  const arr = new Int32Array(padded);
  arr.set(tokens.slice(0, Math.min(length, padded)));
  return { tokens: arr, offset, length };
}

// Small seeded PRNG for the example order (mlx_lm uses numpy's).
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function* order(n, rand) {
  for (;;) {
    const idx = [...Array(n).keys()];
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    yield* idx;
  }
}

function writeAdapters(engine, dir, config) {
  fs.mkdirSync(dir, { recursive: true });
  engine.loraSave(path.join(dir, 'adapters.safetensors'));
  fs.writeFileSync(path.join(dir, 'adapter_config.json'), JSON.stringify(config, null, 4));
}

// Trains adapters on chat examples ({ messages, tools? }). Options follow the
// mlx_lm lora names and defaults. onReport gets progress every stepsPerReport.
// Pass an AbortSignal to stop; the adapters so far are saved.
async function trainLora(chat, data, options = {}) {
  const o = {
    iters: 1000,
    learningRate: 1e-5,
    numLayers: 16,
    rank: 8,
    scale: 20.0,
    maskPrompt: false,
    maxSeqLength: 2048,
    seed: 0,
    gradCheckpoint: false,
    stepsPerReport: 10,
    saveEvery: 100,
    resumeAdapterFile: null,
    valData: [],
    ...options,
  };
  if (!o.adapterPath) throw new Error('adapterPath is required');
  if (!data.length) throw new Error('No training examples');
  const engine = chat.model.engine;
  const config = {
    fine_tune_type: 'lora',
    model: chat.model.dir,
    num_layers: o.numLayers,
    lora_parameters: { rank: o.rank, dropout: 0.0, scale: o.scale },
    learning_rate: o.learningRate,
    iters: o.iters,
    max_seq_length: o.maxSeqLength,
    mask_prompt: o.maskPrompt,
    grad_checkpoint: o.gradCheckpoint,
    seed: o.seed,
    batch_size: 1,
    optimizer: 'adam',
    engine: 'native',
  };

  engine.loraInit({ numLayers: o.numLayers, rank: o.rank, scale: o.scale, seed: o.seed });
  if (o.resumeAdapterFile) engine.loraLoad(o.resumeAdapterFile, o.scale);

  const examples = data.map((d) => padExample(tokenizeExample(chat, d, o.maskPrompt), o.maxSeqLength));
  const valExamples = o.valData.map((d) => padExample(tokenizeExample(chat, d, o.maskPrompt), o.maxSeqLength));
  const next = order(examples.length, mulberry32(o.seed));

  const valLoss = () => {
    if (!valExamples.length) return null;
    let sum = 0;
    let n = 0;
    for (const e of valExamples) {
      const r = engine.loraLoss(e.tokens, e.offset, e.length);
      sum += r.loss * r.tokens;
      n += r.tokens;
    }
    return sum / n;
  };

  let window = { loss: 0, steps: 0, tokens: 0, start: Date.now() };
  let it = 0;
  for (it = 1; it <= o.iters; it++) {
    if (o.signal?.aborted) break;
    const e = examples[next.next().value];
    const r = engine.trainStep(e.tokens, e.offset, e.length, { learningRate: o.learningRate, gradCheckpoint: o.gradCheckpoint });
    window.loss += r.loss;
    window.steps += 1;
    window.tokens += r.tokens;
    if (it % o.stepsPerReport === 0 || it === o.iters) {
      const secs = (Date.now() - window.start) / 1000;
      o.onReport?.({
        iteration: it,
        trainLoss: window.loss / window.steps,
        valLoss: valLoss(),
        tokensPerSecond: window.tokens / secs,
        peakMemory: engine.memory().peak,
      });
      window = { loss: 0, steps: 0, tokens: 0, start: Date.now() };
    }
    if (it % o.saveEvery === 0) writeAdapters(engine, o.adapterPath, config);
    // Let the event loop run (cancel messages, logs) between steps.
    await new Promise((resolve) => setImmediate(resolve));
  }
  writeAdapters(engine, o.adapterPath, config);
  return { iterations: Math.min(it, o.iters), adapterPath: o.adapterPath };
}

module.exports = { trainLora, tokenizeExample, padExample };
