// JS tokenizer + chat template vs transformers (as used by mlx-lm) on the tiny model.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { ChatModel } = require('../chat.js');

const dir = process.env.TINY_MODEL;

test('encodes and renders the chat template like transformers', { skip: !dir && 'TINY_MODEL not set' }, () => {
  const ref = JSON.parse(fs.readFileSync(path.join(dir, 'reference_tokenizer.json'), 'utf8'));
  const chat = new ChatModel(dir);
  for (const { text, ids } of ref.texts) assert.deepStrictEqual(chat.encode(text), ids, text);
  const prompt = chat.prompt(ref.chat);
  assert.strictEqual(prompt, ref.prompt);
  assert.deepStrictEqual(chat.encode(prompt), ref.prompt_ids);
});

test('streams text pieces that join into the full decode', { skip: !dir && 'TINY_MODEL not set' }, () => {
  const chat = new ChatModel(dir);
  chat.model.eos = new Set();
  const messages = [{ role: 'user', content: 'Hola' }];
  const pieces = [...chat.stream(messages, { maxTokens: 20 })];
  const ids = [...chat.model.generate(chat.encode(chat.prompt(messages)), { maxTokens: 20 })];
  assert.strictEqual(pieces.join(''), chat.tokenizer.decode(ids, { skip_special_tokens: false }));
});
