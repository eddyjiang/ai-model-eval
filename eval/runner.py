"""
Async runner: generates all (model, group, template, trial) combinations,
calls each model's API with per-provider rate limiting, and saves raw
responses to SQLite. Skips combinations already present in the DB.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("LITELLM_LOG", "ERROR")
import litellm
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
from tqdm.asyncio import tqdm

from eval import storage
from eval.utils import build_prompt, load_config

litellm.drop_params = True


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


@dataclass
class PromptTask:
    run_id: str
    model_key: str
    model_name: str
    model_cfg: dict
    group_category: str
    group_name: str
    template_id: str
    template_category: str
    template_valence: str
    trial: int
    prompt: str


async def _call_model(task: PromptTask, rate_limiter: RateLimiter, max_retries: int = 3) -> None:
    for attempt in range(max_retries):
        t0 = time.monotonic()
        try:
            await rate_limiter.acquire()
            result = await litellm.acompletion(
                model=task.model_name,
                messages=[{"role": "user", "content": task.prompt}],
                temperature=task.model_cfg["temperature"],
                max_tokens=task.model_cfg["max_tokens"],
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            response_text = result.choices[0].message.content
            usage = getattr(result, "usage", None)
            storage.save_response(
                run_id=task.run_id,
                model_key=task.model_key,
                model_name=task.model_name,
                group_category=task.group_category,
                group_name=task.group_name,
                template_id=task.template_id,
                template_category=task.template_category,
                template_valence=task.template_valence,
                trial=task.trial,
                prompt=task.prompt,
                response=response_text,
                error=None,
                response_time_ms=elapsed_ms,
                input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            )
            return

        except litellm.RateLimitError:
            await asyncio.sleep(2 ** (attempt + 1))

        except litellm.ContentPolicyViolationError as e:
            # API-level content refusal — record as a hard refusal response
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            storage.save_response(
                run_id=task.run_id,
                model_key=task.model_key,
                model_name=task.model_name,
                group_category=task.group_category,
                group_name=task.group_name,
                template_id=task.template_id,
                template_category=task.template_category,
                template_valence=task.template_valence,
                trial=task.trial,
                prompt=task.prompt,
                response=None,
                error=f"content_policy: {e}",
                response_time_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
            )
            return

        except Exception as e:
            if attempt == max_retries - 1:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                storage.save_response(
                    run_id=task.run_id,
                    model_key=task.model_key,
                    model_name=task.model_name,
                    group_category=task.group_category,
                    group_name=task.group_name,
                    template_id=task.template_id,
                    template_category=task.template_category,
                    template_valence=task.template_valence,
                    trial=task.trial,
                    prompt=task.prompt,
                    response=None,
                    error=str(e),
                    response_time_ms=None,
                    input_tokens=None,
                    output_tokens=None,
                )
            else:
                await asyncio.sleep(2 ** attempt)


def build_tasks(
    run_id: str,
    groups_cfg: dict,
    templates_cfg: dict,
    models_cfg: dict,
    model_keys: list[str],
    template_ids: list[str],
    group_names: list[str],
    n_trials: int,
    skip_existing: bool = True,
) -> list[PromptTask]:
    # Build a reverse lookup: group_name → category
    name_to_category: dict[str, str] = {}
    for cat_key, cat_data in groups_cfg["groups"].items():
        for name in cat_data["members"]:
            name_to_category[name] = cat_data["category"]

    tasks: list[PromptTask] = []
    for model_key in model_keys:
        model_cfg = models_cfg["models"][model_key]
        for template_id in template_ids:
            tmpl = templates_cfg["templates"][template_id]
            for group_name in group_names:
                for trial in range(1, n_trials + 1):
                    if skip_existing and storage.response_exists(
                        run_id, model_key, group_name, template_id, trial
                    ):
                        continue
                    tasks.append(
                        PromptTask(
                            run_id=run_id,
                            model_key=model_key,
                            model_name=model_cfg["litellm_model"],
                            model_cfg=model_cfg,
                            group_category=name_to_category.get(group_name, "unknown"),
                            group_name=group_name,
                            template_id=template_id,
                            template_category=tmpl["category"],
                            template_valence=tmpl["valence"],
                            trial=trial,
                            prompt=build_prompt(tmpl["template"], group_name),
                        )
                    )
    return tasks


async def run_async(
    run_id: str,
    model_keys: list[str],
    template_ids: list[str],
    group_names: list[str],
    n_trials: int,
) -> dict:
    groups_cfg, templates_cfg, models_cfg = load_config()
    storage.create_tables()

    tasks = build_tasks(
        run_id, groups_cfg, templates_cfg, models_cfg,
        model_keys, template_ids, group_names, n_trials,
    )

    if not tasks:
        print("All combinations already present in DB — nothing to run.")
        return {"total": 0, "completed": 0, "errors": 0}

    print(f"Running {len(tasks)} prompts across {len(model_keys)} model(s)...")

    # One rate limiter per model (provider)
    rate_limiters: dict[str, RateLimiter] = {
        mk: RateLimiter(models_cfg["models"][mk]["requests_per_minute"])
        for mk in model_keys
    }
    # Semaphore caps active coroutines per model so the event loop isn't
    # flooded with thousands of waiters when tasks >> RPM.
    semaphores: dict[str, asyncio.Semaphore] = {
        mk: asyncio.Semaphore(8)
        for mk in model_keys
    }

    async def run_task(task: PromptTask):
        async with semaphores[task.model_key]:
            await _call_model(task, rate_limiters[task.model_key])

    await tqdm.gather(*[run_task(t) for t in tasks], desc="Prompts")

    stats = storage.get_run_stats(run_id)
    return stats


def run(
    run_id: str,
    model_keys: list[str],
    template_ids: list[str],
    group_names: list[str],
    n_trials: int,
) -> dict:
    return asyncio.run(run_async(run_id, model_keys, template_ids, group_names, n_trials))
