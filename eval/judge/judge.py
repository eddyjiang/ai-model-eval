import asyncio
import logging
import os
import time
from typing import Optional

os.environ.setdefault("LITELLM_LOG", "ERROR")
import litellm
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

from eval.judge.rubric import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from eval.utils import extract_json

litellm.drop_params = True  # ignore unsupported params per provider


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self._interval = 60.0 / max(requests_per_minute, 1)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class Judge:
    def __init__(self, judge_model_key: str, models_config: dict):
        cfg = models_config["models"][judge_model_key]
        self.judge_model_key = judge_model_key
        self.judge_model_name = cfg["litellm_model"]
        self.display_name = cfg["display_name"]
        self._rate_limiter = RateLimiter(cfg["requests_per_minute"])

    async def score(
        self,
        prompt: str,
        group_name: str,
        response: str,
        max_retries: int = 3,
    ) -> tuple[dict, str, Optional[str]]:
        """
        Returns (score_dict, raw_judge_text, error_str_or_None).
        score_dict is empty on error.
        """
        judge_prompt = build_judge_prompt(prompt, group_name, response)

        raw = ""
        for attempt in range(max_retries):
            try:
                await self._rate_limiter.acquire()
                result = await litellm.acompletion(
                    model=self.judge_model_name,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=0,
                    max_tokens=1024,
                )
                raw = result.choices[0].message.content or ""
                score = extract_json(raw)
                return score, raw, None

            except litellm.RateLimitError:
                delay = 2 ** (attempt + 1)
                await asyncio.sleep(delay)

            except Exception as e:
                if attempt == max_retries - 1:
                    return {}, raw, str(e)
                await asyncio.sleep(2 ** attempt)

        return {}, raw, "max retries exceeded"


# Per-judge-model singleton registry, keyed by (judge_model_key, models_config_id)
_judges: dict[str, Judge] = {}


def get_judge(judge_model_key: str, models_config: dict) -> Judge:
    if judge_model_key not in _judges:
        _judges[judge_model_key] = Judge(judge_model_key, models_config)
    return _judges[judge_model_key]
