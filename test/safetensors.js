// Minimal safetensors reader for tests (float32/bfloat16 tensors).
const fs = require('node:fs');

function readSafetensors(file) {
  const buf = fs.readFileSync(file);
  const n = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + n).toString('utf8'));
  const base = 8 + n;
  const out = {};
  for (const [name, t] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [start, end] = t.data_offsets;
    const bytes = buf.subarray(base + start, base + end);
    let values;
    if (t.dtype === 'F32') {
      values = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
    } else if (t.dtype === 'BF16') {
      values = new Float32Array(bytes.length / 2);
      const u = new Uint32Array(values.buffer);
      for (let i = 0; i < values.length; i++) u[i] = bytes.readUInt16LE(2 * i) << 16;
    } else {
      throw new Error(`dtype ${t.dtype} not supported`);
    }
    out[name] = { shape: t.shape, dtype: t.dtype, values };
  }
  return out;
}

module.exports = { readSafetensors };
