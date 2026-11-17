// Tokenizer + chat template in JS (no Python): turns messages into token ids
// and streams decoded text from the native engine.
const fs = require('node:fs');
const path = require('node:path');
const { Tokenizer } = require('@huggingface/tokenizers');
const { Template } = require('@huggingface/jinja');
const { NativeModel } = require('@celticopetit/mlx-node');

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

class ChatModel {
  constructor(modelDir) {
    this.model = new NativeModel(modelDir);
    const config = readJson(path.join(modelDir, 'tokenizer_config.json'));
    this.tokenizer = new Tokenizer(readJson(path.join(modelDir, 'tokenizer.json')), config);
    const templateFile = path.join(modelDir, 'chat_template.jinja');
    const source = fs.existsSync(templateFile) ? fs.readFileSync(templateFile, 'utf8') : config.chat_template;
    if (!source) throw new Error('El model no té plantilla de xat');
    this.template = new Template(source);
    const tokenText = (t) => (typeof t === 'string' ? t : t?.content ?? '');
    this.specials = { bos_token: tokenText(config.bos_token), eos_token: tokenText(config.eos_token) };
  }

  // Same inputs as apply_chat_template in Python: messages, tools and any
  // template variables (e.g. enable_thinking).
  prompt(messages, { tools, ...vars } = {}) {
    return this.template.render({ messages, tools, add_generation_prompt: true, ...this.specials, ...vars });
  }

  encode(text) {
    return this.tokenizer.encode(text, { add_special_tokens: false }).ids;
  }

  // Yields text pieces as they are generated. Options go to the engine
  // (maxTokens, temperature, topP, topK, seed).
  *stream(messages, options = {}) {
    const { tools, templateVars, ...sampling } = options;
    yield* this.streamIds(this.encode(this.prompt(messages, { tools, ...templateVars })), sampling);
  }

  // The same from prompt token ids.
  *streamIds(ids, sampling = {}) {
    let pending = [];
    let shown = '';
    for (const id of this.model.generate(ids, sampling)) {
      pending.push(id);
      const text = this.tokenizer.decode(pending, { skip_special_tokens: false });
      // A token can end in the middle of a UTF-8 character; wait for the rest.
      if (text.endsWith('�')) continue;
      if (text.length > shown.length) yield text.slice(shown.length);
      shown = text;
      // Restart the window at line ends so decoding stays cheap.
      if (text.endsWith('\n')) {
        pending = [];
        shown = '';
      }
    }
    // Generation ended mid-character: show what is left, as decode() does.
    const text = this.tokenizer.decode(pending, { skip_special_tokens: false });
    if (text.length > shown.length) yield text.slice(shown.length);
  }
}

module.exports = { ChatModel };
