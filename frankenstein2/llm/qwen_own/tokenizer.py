"""The checkpoint's tokenizer and chat template without transformers: `tokenizers` (tokenizer.json) + jinja2 for the
chat template (chat_template.jinja, or tokenizer_config.json's "chat_template"), rendered with the same environment
options HF uses (trim_blocks, lstrip_blocks, loop controls, `tojson` with ensure_ascii=False, raise_exception,
strftime_now). Interface used by the engine: apply_chat_template / decode / convert_tokens_to_ids / encode."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class QwenTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        path = Path(path)
        self.tk = Tokenizer.from_file(str(path / "tokenizer.json"))
        cfg = json.loads((path / "tokenizer_config.json").read_text()) if (path / "tokenizer_config.json").exists() else {}
        tpl = path / "chat_template.jinja"
        self.template_src = tpl.read_text() if tpl.exists() else cfg.get("chat_template")
        if isinstance(self.template_src, list):                    # [{"name": "default", "template": ...}, ...]
            self.template_src = next(t["template"] for t in self.template_src if t.get("name") == "default")
        self.special = {k: cfg.get(k) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
        for k, v in list(self.special.items()):
            if isinstance(v, dict):
                self.special[k] = v.get("content")
        self._tpl = None

    def _template(self):
        if self._tpl is None:
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def raise_exception(msg):
                raise jinja2.exceptions.TemplateError(msg)

            def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
                return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=["jinja2.ext.loopcontrols"])
            env.filters["tojson"] = tojson
            env.globals["raise_exception"] = raise_exception
            env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
            self._tpl = env.from_string(self.template_src)
        return self._tpl

    def render(self, messages, tools=None, add_generation_prompt=True, **kw) -> str:
        return self._template().render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                       **self.special, **kw)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True, return_dict=False,
                            **kw):
        text = self.render(messages, tools=tools or None, add_generation_prompt=add_generation_prompt, **kw)
        return self.encode(text) if tokenize else text

    def encode(self, text: str) -> list:
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=False) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, token: str):
        return self.tk.token_to_id(token)
