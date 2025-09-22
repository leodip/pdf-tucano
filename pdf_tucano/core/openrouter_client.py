from __future__ import annotations

import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional

import backoff
import requests

from pdf_tucano.config import get_settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Thin wrapper around the OpenRouter chat completions API."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    GENERATION_URL = "https://openrouter.ai/api/v1/generation"
    GENERATIONS_PATH_URL = "https://openrouter.ai/api/v1/generations"
    GENERATION_RETRY_DELAYS = (0.0, 1.0, 2.5, 4.0, 6.0)

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        logger.info(
            "OpenRouter client initialized (model=%s, probe_enabled=%s)",
            self.model,
            False,
        )

    @staticmethod
    def _log_backoff(details: Dict[str, Any]) -> None:
        logger.warning(
            "Retrying OpenRouter request",
            extra={
                "attempt": details.get("tries"),
                "wait_seconds": round(details.get("wait", 0), 2),
                "target": getattr(details.get("target"), "__name__", "unknown"),
            },
        )

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        on_backoff=lambda details: OpenRouterClient._log_backoff(details),
    )
    def convert_image_to_markdown(self, base64_image: str) -> "OpenRouterCompletion":
        """Send an image to OpenRouter and receive markdown response."""
        system_message = (
            "You are an expert in reading document images and producing faithful, well-structured "
            "Markdown that preserves layout, semantics, numeric accuracy, and visual context. "
            "Always reply with pure Markdown content and never wrap it in code fences or triple "
            "backticks. Ensure any bracketed descriptions match the language of the nearby text."
        )
        user_prompt = (
            "Analyze the attached PDF page image and return all meaningful content as Markdown in the "
            "language(s) used on the page. Do not translate.\n\n"
            "While extracting the content:\n"
            "- Preserve the heading hierarchy (H1, H2, etc.) and overall document structure\n"
            "- Keep bullet, numbered, and nested lists intact\n"
            "- Reproduce tables using Markdown table syntax, keeping column order and numeric formatting\n"
            "- Treat numeric data as critical; copy numbers, currencies, percentages, and dates exactly as they appear and double-check transcription accuracy\n"
            "- Capture text from diagrams, charts, callouts, stamps, and annotations; describe non-textual visuals in [brackets] using the same language as the surrounding content\n"
            "- Maintain emphasis (bold/italic/underline), inline math, and paragraph breaks\n"
            "- Include figure captions, footnotes, and references associated with visual elements\n"
            "- If the complete content would exceed the 8192 token limit, provide a concise summary only for the overflow while keeping critical data\n\n"
            "General requirements:\n"
            "- Important: return only the Markdown content. Do not include triple backticks, code fences, language identifiers, or any line that starts with ``` or similar fence markers; if you would have added them, remove them entirely.\n"
            "- Support multilingual pages (e.g., Portuguese, French, German); keep each section in its original language\n"
            "- All content in [brackets] must use the same language as the surrounding text; never leave bracketed descriptions in English when the neighboring prose is in another language (e.g., use '[Imagem de …]' for Portuguese content)\n"
            "- Numeric fidelity is paramount; never invent, round, or translate numbers, and flag uncertainties in [verificar] while keeping the original value\n"
            "- Represent equations using LaTeX inside `$...$` or `$$...$$` as appropriate\n"
            "- When precise formatting is unclear, make the best reasonable Markdown approximation and mention uncertainties in [brackets]"
        )

        payload = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                },
            ],
        }

        call_timer = time.perf_counter()
        response = self.session.post(self.OPENROUTER_URL, json=payload, timeout=120)
        duration_seconds = round(time.perf_counter() - call_timer, 3)
        response.raise_for_status()
        logger.debug(
            "OpenRouter convert request completed",
            extra={
                "model": self.model,
                "status_code": response.status_code,
                "duration_seconds": duration_seconds,
                "response_bytes": len(response.content) if response.content else 0,
            },
        )
        data = response.json()

        try:
            markdown = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError("Unexpected OpenRouter response structure") from exc
        return OpenRouterCompletion(markdown=markdown, generation_id=data.get("id"))

    def fetch_generation_stats(self, generation_id: str) -> "GenerationStats":
        """Retrieve native token usage and cost details for a generation."""
        if not generation_id:
            raise ValueError("Generation id is required to fetch stats")

        urls = [
            (self.GENERATION_URL, {"params": {"id": generation_id}}),
            (f"{self.GENERATIONS_PATH_URL}/{generation_id}", {}),
        ]

        last_error: Optional[Exception] = None
        last_duration: Optional[float] = None
        total_attempts = len(self.GENERATION_RETRY_DELAYS)

        for attempt in range(total_attempts):
            attempt_number = attempt + 1
            attempts_left = total_attempts - attempt_number

            logger.debug(
                "Attempting generation stats fetch",
                extra={
                    "generation_id": generation_id,
                    "attempt": attempt_number,
                    "total_attempts": total_attempts,
                },
            )

            for url, request_kwargs in urls:
                request_timer = time.perf_counter()
                try:
                    response = self.session.get(url, timeout=30, **request_kwargs)
                except requests.RequestException as req_exc:
                    last_duration = round(time.perf_counter() - request_timer, 3)
                    logger.warning(
                        "Generation stats request error",
                        extra={
                            "generation_id": generation_id,
                            "url": url,
                            "attempt": attempt_number,
                            "total_attempts": total_attempts,
                            "error": str(req_exc),
                            "duration_seconds": last_duration,
                        },
                    )
                    last_error = req_exc
                    continue

                last_duration = round(time.perf_counter() - request_timer, 3)
                status = response.status_code
                body_preview = response.text[:500] if response.text else None

                if status >= 400:
                    logger.warning(
                        "Generation stats request failed",
                        extra={
                            "generation_id": generation_id,
                            "url": url,
                            "status": status,
                            "attempt": attempt_number,
                            "total_attempts": total_attempts,
                            "body_preview": body_preview,
                            "duration_seconds": last_duration,
                        },
                    )
                    last_error = requests.HTTPError(response=response)
                    if status == 404:
                        next_delay = (
                            self.GENERATION_RETRY_DELAYS[attempt + 1]
                            if attempts_left > 0
                            else None
                        )
                        logger.debug(
                            "Generation stats 404; will retry if attempts remain",
                            extra={
                                "generation_id": generation_id,
                                "url": url,
                                "attempt": attempt_number,
                                "attempts_left": attempts_left,
                                "next_delay": next_delay,
                                "duration_seconds": last_duration,
                            },
                        )
                        continue
                    response.raise_for_status()

                try:
                    payload = response.json()
                except ValueError as json_error:
                    logger.warning(
                        "Generation stats response not JSON (id=%s url=%s attempt=%s body=%s error=%s)",
                        generation_id,
                        url,
                        attempt_number,
                        body_preview,
                        str(json_error),
                        extra={"duration_seconds": last_duration},
                    )
                    last_error = json_error
                    continue

                if self._payload_indicates_missing(payload):
                    logger.warning(
                        "Generation stats payload reported missing generation (id=%s url=%s attempt=%s payload=%s)",
                        generation_id,
                        url,
                        attempt_number,
                        str(payload)[:500],
                        extra={"duration_seconds": last_duration},
                    )
                    last_error = RuntimeError("Generation stats payload indicates missing generation")
                    continue

                logger.debug(
                    "Generation stats payload received (id=%s url=%s attempt=%s has_data=%s)",
                    generation_id,
                    url,
                    attempt_number,
                    isinstance(payload, dict) and bool(payload.get("data")),
                    extra={"duration_seconds": last_duration},
                )

                record = self._extract_generation_record(payload)
                if record is None:
                    logger.warning(
                        "Generation stats missing data (id=%s url=%s attempt=%s payload_keys=%s)",
                        generation_id,
                        url,
                        attempt_number,
                        list(payload.keys()) if isinstance(payload, dict) else "non-dict",
                        extra={"duration_seconds": last_duration},
                    )
                    last_error = RuntimeError("Generation stats response missing data")
                    continue

                stats = self._parse_generation_record(record, generation_id)

                logger.info(
                    "Generation stats retrieved",
                    extra={
                        "generation_id": generation_id,
                        "url": url,
                        "attempt": attempt_number,
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "total_tokens": stats.total_tokens,
                        "total_cost": str(stats.total_cost) if stats.total_cost is not None else None,
                        "currency": stats.currency,
                        "duration_seconds": last_duration,
                    },
                )

                if stats.total_cost is None:
                    logger.info(
                        "Generation reported zero or missing cost (id=%s)",
                        generation_id,
                        extra={"duration_seconds": last_duration},
                    )

                return stats

            if attempts_left > 0:
                next_delay = self.GENERATION_RETRY_DELAYS[attempt + 1]
                logger.info(
                    "Generation stats retry scheduled",
                    extra={
                        "generation_id": generation_id,
                        "next_attempt": attempt_number + 1,
                        "total_attempts": total_attempts,
                        "attempts_left": attempts_left,
                        "wait_seconds": next_delay,
                        "last_error": str(last_error) if last_error else None,
                        "last_duration_seconds": last_duration,
                    },
                )
                if next_delay:
                    time.sleep(next_delay)

        if last_error:
            if isinstance(last_error, requests.HTTPError) and last_error.response is not None and last_error.response.status_code == 404:
                logger.error(
                    "OpenRouter never published generation stats",
                    extra={
                        "generation_id": generation_id,
                        "last_duration_seconds": last_duration,
                    },
                )
            raise last_error

        raise RuntimeError("Unable to retrieve generation stats")

    @staticmethod
    def _payload_indicates_missing(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if str(code) == "404":
                return True
            message = str(error.get("message", "")).lower()
            if "not found" in message:
                return True
        return False

    @staticmethod
    def _extract_generation_record(payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None

        data = payload.get("data")
        if isinstance(data, list):
            if not data:
                return None
            first = data[0]
            return first if isinstance(first, dict) else None
        if isinstance(data, dict):
            return data

        # Some providers may return the record at the top level
        return payload if payload.get("id") else None

    @staticmethod
    def _parse_generation_record(record: Dict[str, Any], generation_id: str) -> "GenerationStats":
        raw_usage = record.get("usage") or record.get("native_usage")
        usage: Dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        usage_total_value = raw_usage if raw_usage is not None and not isinstance(raw_usage, dict) else None

        prompt_tokens = OpenRouterClient._to_int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt")
            or record.get("tokens_prompt")
            or record.get("native_tokens_prompt")
        )
        completion_tokens = OpenRouterClient._to_int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("completion")
            or record.get("tokens_completion")
            or record.get("native_tokens_completion")
        )
        total_tokens = OpenRouterClient._to_int(
            usage.get("total_tokens")
            or usage.get("total")
            or record.get("native_tokens_total")
        )
        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        if total_tokens is None and usage_total_value is not None:
            total_tokens = OpenRouterClient._to_int(usage_total_value)

        pricing_candidates = [
            record.get("pricing"),
            record.get("cost"),
            record.get("costs"),
        ]

        pricing: Dict[str, Any] = {}
        direct_total_cost: Optional[Decimal] = None

        for candidate in pricing_candidates:
            if candidate is None:
                continue
            if isinstance(candidate, dict):
                pricing = candidate
                break
            if isinstance(candidate, list):
                dict_candidate = next(
                    (item for item in candidate if isinstance(item, dict)),
                    None,
                )
                if dict_candidate is not None:
                    pricing = dict_candidate
                    break
                continue
            if direct_total_cost is None:
                direct_total_cost = OpenRouterClient._to_decimal(candidate)

        input_cost = OpenRouterClient._to_decimal(
            pricing.get("input")
            or pricing.get("input_cost")
            or pricing.get("prompt")
            or record.get("input_cost")
        )
        output_cost = OpenRouterClient._to_decimal(
            pricing.get("output")
            or pricing.get("output_cost")
            or pricing.get("completion")
            or record.get("output_cost")
        )
        total_cost = OpenRouterClient._to_decimal(
            pricing.get("total")
            or pricing.get("total_cost")
            or record.get("total_cost")
        )
        if total_cost is None and direct_total_cost is not None:
            total_cost = direct_total_cost
        if total_cost is None and input_cost is not None and output_cost is not None:
            total_cost = input_cost + output_cost

        currency = (
            pricing.get("currency")
            or pricing.get("unit")
            or record.get("currency")
        )

        return GenerationStats(
            generation_id=generation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            currency=currency,
        )

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (ArithmeticError, ValueError):
            return None


@dataclass
class OpenRouterCompletion:
    markdown: str
    generation_id: Optional[str]


@dataclass
class GenerationStats:
    generation_id: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    input_cost: Optional[Decimal]
    output_cost: Optional[Decimal]
    total_cost: Optional[Decimal]
    currency: Optional[str]
