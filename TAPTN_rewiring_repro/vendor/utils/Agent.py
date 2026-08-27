"""LLM client for TAPTN rewiring re-runs.

Reads credentials from the environment (never from source):

    OPENAI_API_KEY   or TAPTN_LLM_API_KEY
    OPENAI_BASE_URL  (default https://openrouter.ai/api/v1)

Optional OpenRouter-style extra_body for hybrid-reasoning models is preserved
so live re-runs stay in non-thinking / low-reasoning mode, matching the paper.
"""
from __future__ import annotations

import json
import os
from time import sleep
import types

import openai
import tiktoken


def _api_key():
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("TAPTN_LLM_API_KEY") or ""


def _base_url():
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("TAPTN_LLM_BASE_URL") or "https://openrouter.ai/api/v1"


class Agent:
    def __init__(self, name, role, model=None):
        if model is None:
            model = "meta-llama/llama-3.1-8b-instruct"
        self.name = name
        self.role = role
        self.model = model
        self.sys_prompt = f"You are {name}, {role}."

    def count_tokens(self, messages):
        encoding = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        for message in messages:
            total_tokens += len(encoding.encode(message["content"]))
        return total_tokens

    def _client(self):
        key = _api_key()
        if not key:
            raise RuntimeError(
                "No API key. Export OPENAI_API_KEY or TAPTN_LLM_API_KEY "
                "(and optionally OPENAI_BASE_URL) before a live run."
            )
        return openai.OpenAI(api_key=key, base_url=_base_url())

    def get_completion_from_messages(self, messages, model=None, temperature=1, max_tokens=500):
        max_tokens_per_model = {
            "qwen/qwen2.5-vl-72b-instruct": 32000,
            "meta-llama/llama-3.1-8b-instruct": 16 * 1024,
            "google/gemma-3-27b-it": 32 * 1024,
            "minimax/minimax-01": 32 * 1024,
            "qwen/qwq-32b": 32 * 1024,
        }
        if model is None:
            model = self.model

        max_tokens = max_tokens_per_model.get(model, 16 * 1024) - self.count_tokens(messages) - 150
        if "8b" not in model:
            max_tokens = min(16000 - self.count_tokens(messages) - 1500, 4096)
        if -5000 < max_tokens < 1000:
            max_tokens = 1000
        if max_tokens < -5000:
            max_tokens = 1000

        extra_body = {}
        if model in ("z-ai/glm-5.1", "qwen/qwen3.5-27b"):
            extra_body = {"reasoning": {"enabled": False}}
        elif model == "openai/gpt-oss-120b":
            extra_body = {"reasoning": {"effort": "low"}}

        client = self._client()
        last_err = None
        for _ in range(3):
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=150,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if isinstance(content, tuple):
                    return content[0]
                return content
            except Exception as e:
                last_err = e
                print(f"Error: {e}")
                sleep(10)
        raise RuntimeError(f"The completion did not finish after 3 times: {last_err}")

    def get_robust_completion(self, messages, description="completion", min_length=10,
                              max_retries=10, temperature=None):
        import concurrent.futures

        response = None
        for attempt in range(max_retries):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                if temperature is not None:
                    future = executor.submit(
                        self.get_completion_from_messages, messages, temperature=temperature
                    )
                else:
                    future = executor.submit(self.get_completion_from_messages, messages)
                try:
                    response = future.result(timeout=180)
                except concurrent.futures.TimeoutError:
                    print(f"Timeout: {description} exceeded 180 seconds. Retrying ({attempt+1}/{max_retries})...")
                    executor.shutdown(wait=False)
                    sleep(2)
                    continue
            except Exception as e:
                print(f"Error in {description} attempt {attempt+1}: {e}")
                executor.shutdown(wait=False)
                sleep(5)
                continue
            finally:
                executor.shutdown(wait=False)

            if response is not None and (len(response) >= min_length or "N/A" in response):
                return response
            print(f"Warning: {description} response too short. Retrying ({attempt+1}/{max_retries})...")
            sleep(1)

        if response:
            print(f"Warning: Returning short {description} after {max_retries} attempts")
            return response
        fallback = f"[Unable to generate {description} after {max_retries} attempts]"
        print(f"Error: {fallback}")
        return fallback
