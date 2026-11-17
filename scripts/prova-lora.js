// Trains LoRA adapters with the native engine on a real model and measures it.
//   node scripts/prova-lora.js <modelDir> <train.jsonl> <adapterDir> [--iters 10]
//        [--lr 1e-5] [--layers 16] [--max-seq 4096] [--mask-prompt] [--grad-checkpoint]
// train.jsonl: one {"messages": [...], "tools"?: [...]} per line (mlx_lm chat format).
// The adapters load in mlx-lm and in the native engine (NativeModel.loadAdapters).
const fs = require('node:fs');
const { ChatModel } = require('../chat.js');
const { trainLora } = require('../lora.js');

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const [modelDir, dataFile, adapterPath] = args.filter((a, i) => !a.startsWith('--') && !['--iters', '--lr', '--layers', '--max-seq'].includes(args[i - 1]));
if (!modelDir || !dataFile || !adapterPath) {
  console.error('Ús: node scripts/prova-lora.js <model> <train.jsonl> <carpeta adaptadors> [--iters 10] [--mask-prompt] [--grad-checkpoint]');
  process.exit(1);
}
const data = fs.readFileSync(dataFile, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));

(async () => {
  let t0 = Date.now();
  const chat = new ChatModel(modelDir);
  console.log(`Model carregat en ${((Date.now() - t0) / 1000).toFixed(1)} s, ${data.length} exemples`);
  t0 = Date.now();
  await trainLora(chat, data, {
    adapterPath,
    iters: Number(opt('--iters', 10)),
    learningRate: Number(opt('--lr', 1e-5)),
    numLayers: Number(opt('--layers', 16)),
    maxSeqLength: Number(opt('--max-seq', 4096)),
    maskPrompt: args.includes('--mask-prompt'),
    gradCheckpoint: args.includes('--grad-checkpoint'),
    stepsPerReport: 1,
    onReport: (r) => console.log(`iter ${r.iteration}: loss ${r.trainLoss.toFixed(3)}, ${r.tokensPerSecond.toFixed(0)} tok/s, pic ${(r.peakMemory / 2 ** 30).toFixed(2)} GB`),
  });
  console.log(`Fet en ${((Date.now() - t0) / 60000).toFixed(1)} min. Adaptadors a ${adapterPath}`);
})();
