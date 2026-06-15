import json
import logging
import time

from openai import APIError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


class LLMProcessor:
    MAX_BACKOFF_SECONDS = 120.0

    def __init__(
        self,
        model_name: str,
        base_url: str,
        use_json_mode: bool = True,
        temperature: float = 0.1,
        max_retries: int = 5,
        retry_backoff_factor: float = 5.0,
        request_timeout: float = 120.0,
    ):
        self.model_name = model_name
        self.use_json_mode = use_json_mode
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.request_timeout = request_timeout

        self.client = OpenAI(base_url=base_url, timeout=request_timeout)

    def _ask_llm(self, system_prompt: str, user_content: str) -> list[dict]:
        retries = 0
        backoff_factor = self.retry_backoff_factor

        while retries < self.max_retries:
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                            + "\nOutput strictly valid JSON with a 'data' key containing the list of objects.",
                        },
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": self.temperature,
                }

                if self.use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                raw_content = response.choices[0].message.content.strip()

                try:
                    result = json.loads(raw_content)
                except json.JSONDecodeError:
                    cleaned = raw_content.replace("```json", "").replace("```", "").strip()
                    try:
                        result = json.loads(cleaned)
                    except json.JSONDecodeError:
                        logger.error(f"LLM returned invalid JSON (first 300 chars): {raw_content[:300]!r}")
                        return []

                if isinstance(result, list):
                    return result

                if isinstance(result, dict):
                    if "data" in result and isinstance(result["data"], list):
                        return result["data"]

                    list_values = [v for v in result.values() if isinstance(v, list)]
                    if len(list_values) == 1:
                        return list_values[0]

                    logger.warning(f"Unexpected JSON structure with {len(list_values)} list keys: {list(result.keys())}")

                return []

            except RateLimitError:
                logger.warning(
                    f"API limit reached (429). Waiting {backoff_factor:.0f}s before attempt {retries + 1}/{self.max_retries}..."
                )
                time.sleep(backoff_factor)
                retries += 1
                backoff_factor = min(backoff_factor * 2, self.MAX_BACKOFF_SECONDS)

            except APIError as e:
                logger.warning(
                    f"Temporary API server failure (possibly 503). Waiting {backoff_factor:.0f}s before attempt {retries + 1}/{self.max_retries}. Details: {e}"
                )
                time.sleep(backoff_factor)
                retries += 1
                backoff_factor = min(backoff_factor * 2, self.MAX_BACKOFF_SECONDS)

            except Exception:
                logger.exception("Critical LLM or network error")
                return []

        logger.error(f"Maximum retry count exceeded ({self.max_retries}) because of API limits or failures. Skipping chunk.")
        return []

    def extract_characters(self, text: str, prompt: str) -> list[dict]:
        return self._ask_llm(prompt, text)

    def extract_relations(self, text: str, characters: list, prompt: str) -> list[dict]:
        user_content = (
            f"DOCUMENT 1 (Text):\n{text}\n\nDOCUMENT 2 (Characters):\n{json.dumps(characters, ensure_ascii=False)}"
        )
        return self._ask_llm(prompt, user_content)

    def extract_locations(self, text: str, prompt: str) -> list[dict]:
        return self._ask_llm(prompt, text)

    def extract_time(self, text: str, prompt: str) -> list[dict]:
        return self._ask_llm(prompt, text)
