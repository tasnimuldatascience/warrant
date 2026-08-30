"""The local generator.

Small and local on purpose. Everything here has to run from a clean clone on one laptop with
no API key and no account, so the default is Qwen2.5-1.5B-Instruct at roughly 3 GB in fp16 --
which co-exists with the 130 MB embedder and the 90 MB cross-encoder inside 8 GB of VRAM with
room to spare.

That is a real ceiling and it is not hidden: generation is the last stage of the pipeline, so
whatever the model gets wrong shows up as its own row of the failure budget rather than as a
number nobody can attribute. Swapping in a larger model is a config change, and the budget
says what it bought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .answer import (
    DEFAULT_MODEL,
    MAX_NEW_TOKENS,
    Answer,
    build_prompt,
    ground,
    parse_response,
)

#: Loaded once per process. A generator constructed per call would dominate every timing.
_LOADED: dict[str, Any] = {}


def _pipeline(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name in _LOADED:
        return _LOADED[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    _LOADED[model_name] = (tokenizer, model)
    return _LOADED[model_name]


@dataclass
class Generator:
    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = MAX_NEW_TOKENS
    #: Greedy first. The task is extraction and citation, not prose: sampling buys nothing
    #: and costs reproducibility, which a benchmark and a replay both depend on.
    temperature: float = 0.0
    stats: dict[str, int] = field(default_factory=lambda: {"calls": 0, "retries": 0,
                                                           "parse_failures": 0})

    def complete(self, messages: list[dict], *, temperature: float | None = None) -> str:
        import torch

        tokenizer, model = _pipeline(self.model_name)
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        temp = self.temperature if temperature is None else temperature
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=temp > 0, temperature=temp or None,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    def answer(self, question: str, excerpts: list[tuple[str, str, str]], *,
               as_of: str, scope: str) -> Answer:
        """One grounded answer, or an abstention.

        A malformed response is retried once. If the second attempt is also unparseable the
        answer is abandoned rather than salvaged: a half-parsed response yields claims whose
        citations were never really made, which is worse than saying nothing.
        """
        cited = {vid: text for vid, _heading, text in excerpts}
        if not excerpts:
            return Answer(question, as_of, scope, [], False, cited)

        messages = build_prompt(question, excerpts)
        raw = ""
        for attempt in range(2):
            self.stats["calls"] += 1
            if attempt:
                self.stats["retries"] += 1
            raw = self.complete(messages, temperature=0.0)
            parsed = parse_response(raw, excerpts)
            if parsed is not None:
                claims, found = parsed
                return Answer(question, as_of, scope, ground(claims, cited), found,
                              cited, raw=raw)

        self.stats["parse_failures"] += 1
        return Answer(question, as_of, scope, [], False, cited, raw=raw, parse_failed=True)
