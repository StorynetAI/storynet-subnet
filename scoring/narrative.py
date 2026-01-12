"""
Narrative Merit Score Module
============================

Calculates Narrative Merit Score (30 points) using AI-based evaluation.

This module evaluates:
- Storytelling craft and narrative flow
- Emotional resonance and reader engagement
- Creative depth and originality of ideas
- Character voice consistency
- World-building coherence

The evaluation prompts are loaded from server-side config (not in repo),
allowing validators to customize their evaluation criteria.

This is subjective evaluation by design - external miners cannot reverse-engineer
the exact criteria, providing natural protection against gaming.
"""

import json
import os
import hashlib
import time
from typing import Dict, Any, Tuple, Optional, List
import bittensor as bt

# Try to import LLM clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Default evaluation prompt (used if no custom config exists)
# This is intentionally generic - validators should customize
# Note: Use {{}} for literal braces in the template
DEFAULT_EVALUATION_PROMPT = """You are a professional story editor evaluating narrative quality.

Evaluate the following story content on these dimensions:
1. Narrative Flow (0-5): How smoothly does the story progress?
2. Emotional Impact (0-5): Does the story evoke emotions?
3. Creative Originality (0-5): Are the ideas fresh and interesting?
4. Internal Consistency (0-5): Is the story logically coherent?

Story Content:
{content}

Context (if provided):
{context}

Respond with ONLY a JSON object in this format:
{{
    "narrative_flow": <score 0-5>,
    "emotional_impact": <score 0-5>,
    "creative_originality": <score 0-5>,
    "internal_consistency": <score 0-5>,
    "brief_notes": "<1-2 sentence summary>"
}}
"""


class NarrativeEvaluator:
    """
    AI-based narrative quality evaluator.

    Uses OpenAI-compatible API, which works with:
    - OpenAI
    - vLLM
    - SGLang
    - Ollama
    - LocalAI
    - Any OpenAI-compatible endpoint
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            config_path: Path to evaluation config file (YAML/JSON)
                        If None, uses default path or falls back to defaults
        """
        self.config = self._load_config(config_path)
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour cache

        # Statistics
        self.total_evaluations = 0
        self.cache_hits = 0
        self.api_errors = 0

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load evaluation config from file or use defaults."""

        # Try multiple paths
        search_paths = []
        if config_path:
            search_paths.append(config_path)

        # Server-side config paths (NOT in git repo)
        search_paths.extend([
            os.path.expanduser("~/.storynet/narrative_config.yaml"),
            os.path.expanduser("~/.storynet/narrative_config.json"),
            "/etc/storynet/narrative_config.yaml",
            os.path.join(os.path.dirname(__file__), "../config/narrative_eval.yaml"),
        ])

        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        if path.endswith('.yaml') or path.endswith('.yml'):
                            import yaml
                            config = yaml.safe_load(f)
                        else:
                            config = json.load(f)
                        bt.logging.info(f"Loaded narrative config from: {path}")
                        return config
                except Exception as e:
                    bt.logging.warning(f"Failed to load config from {path}: {e}")

        # Return default config
        bt.logging.info("Using default narrative evaluation config")
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "enabled": True,
            "api_base": None,  # Auto-detect from environment
            "api_key": None,   # Auto-detect from environment
            "model": "gpt-4o-mini",
            "timeout": 30,
            "max_retries": 2,
            "evaluation_prompt": DEFAULT_EVALUATION_PROMPT,
            "dimension_weights": {
                "narrative_flow": 0.30,
                "emotional_impact": 0.25,
                "creative_originality": 0.25,
                "internal_consistency": 0.20
            },
            "cache_enabled": True,
            "fallback_score": 15.0  # Middle score if AI fails (0-30 range)
        }

    def _get_content_hash(self, content: str) -> str:
        """Generate hash for caching."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _check_cache(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Check if result is cached."""
        if not self.config.get("cache_enabled", True):
            return None

        if content_hash in self.cache:
            entry = self.cache[content_hash]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                self.cache_hits += 1
                return entry["result"]
            else:
                del self.cache[content_hash]
        return None

    def _cache_result(self, content_hash: str, result: Dict[str, Any]):
        """Cache evaluation result."""
        if self.config.get("cache_enabled", True):
            self.cache[content_hash] = {
                "result": result,
                "timestamp": time.time()
            }
            # Limit cache size
            if len(self.cache) > 1000:
                oldest = min(self.cache.items(), key=lambda x: x[1]["timestamp"])
                del self.cache[oldest[0]]

    def _detect_api_config(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Auto-detect API configuration from environment.

        Returns:
            Tuple of (api_base, api_key)
        """
        # Check config first
        api_base = self.config.get("api_base")
        api_key = self.config.get("api_key")

        if api_base:
            return api_base, api_key or os.getenv("OPENAI_API_KEY", "")

        # Auto-detect from environment
        if os.getenv("OPENAI_API_KEY"):
            return "https://api.openai.com/v1", os.getenv("OPENAI_API_KEY")

        if os.getenv("OPENAI_API_BASE"):
            return os.getenv("OPENAI_API_BASE"), os.getenv("OPENAI_API_KEY", "")

        # Try local endpoints (vLLM, SGLang, Ollama)
        local_endpoints = [
            "http://localhost:8000/v1",   # vLLM default
            "http://localhost:30000/v1",  # SGLang default
            "http://localhost:11434/v1",  # Ollama
        ]

        if HTTPX_AVAILABLE:
            for endpoint in local_endpoints:
                try:
                    with httpx.Client(timeout=2) as client:
                        resp = client.get(f"{endpoint}/models")
                        if resp.status_code == 200:
                            bt.logging.info(f"Auto-detected local LLM at {endpoint}")
                            return endpoint, ""
                except:
                    continue

        return None, None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call LLM using OpenAI-compatible API.

        Works with OpenAI, vLLM, SGLang, Ollama, LocalAI, etc.
        """
        api_base, api_key = self._detect_api_config()

        if not api_base:
            bt.logging.debug("No LLM API configured or detected")
            return None

        model = self.config.get("model", "gpt-4o-mini")
        timeout = self.config.get("timeout", 30)

        # Use OpenAI client if available
        if OPENAI_AVAILABLE:
            try:
                client = openai.OpenAI(
                    base_url=api_base,
                    api_key=api_key or "not-needed",
                    timeout=timeout
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional story editor. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                bt.logging.warning(f"LLM API error: {e}")
                self.api_errors += 1

        # Fallback to httpx
        elif HTTPX_AVAILABLE:
            try:
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        f"{api_base}/chat/completions",
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "You are a professional story editor. Respond only with valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.3,
                            "max_tokens": 500
                        }
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                bt.logging.warning(f"LLM API error: {e}")
                self.api_errors += 1

        return None

    def _parse_ai_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse AI response to extract scores."""
        if not response:
            return None

        # Clean response
        response = response.strip()

        # Try to find JSON in response
        try:
            # Direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in response (greedy match for nested objects)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # More aggressive: find first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                json_str = response[first_brace:last_brace + 1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        return None

    def evaluate(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        task_type: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate narrative quality using AI.

        Args:
            data: Parsed story data
            context: Context (blueprint, user_input, etc.)
            task_type: Type of task

        Returns:
            Tuple of (score 0-30, breakdown dict)
        """
        self.total_evaluations += 1

        breakdown = {
            "narrative_flow": 0.0,
            "emotional_impact": 0.0,
            "creative_originality": 0.0,
            "internal_consistency": 0.0,
            "ai_notes": "",
            "evaluation_method": "ai"
        }

        # Check if enabled
        if not self.config.get("enabled", True):
            breakdown["evaluation_method"] = "disabled"
            return self.config.get("fallback_score", 10.0), breakdown

        # Extract content for evaluation
        content = self._extract_content(data, task_type)
        if not content or len(content) < 50:
            breakdown["evaluation_method"] = "insufficient_content"
            return 5.0, breakdown  # Low score for insufficient content

        # Check cache
        content_hash = self._get_content_hash(content)
        cached = self._check_cache(content_hash)
        if cached:
            breakdown["evaluation_method"] = "cached"
            return cached["score"], cached["breakdown"]

        # Build prompt
        prompt_template = self.config.get("evaluation_prompt", DEFAULT_EVALUATION_PROMPT)
        context_str = json.dumps(context, ensure_ascii=False, indent=2)[:1000]  # Limit context
        prompt = prompt_template.format(content=content[:3000], context=context_str)

        # Call LLM
        response = None
        for attempt in range(self.config.get("max_retries", 2) + 1):
            response = self._call_llm(prompt)
            if response:
                break
            time.sleep(0.5 * (attempt + 1))

        # Parse response
        parsed = self._parse_ai_response(response)

        if not parsed:
            bt.logging.debug("LLM evaluation failed, using fallback score")
            breakdown["evaluation_method"] = "fallback"
            return self.config.get("fallback_score", 15.0), breakdown

        # Extract dimension scores
        weights = self.config.get("dimension_weights", {
            "narrative_flow": 0.30,
            "emotional_impact": 0.25,
            "creative_originality": 0.25,
            "internal_consistency": 0.20
        })

        total_score = 0.0
        for dim, weight in weights.items():
            dim_score = float(parsed.get(dim, 2.5))
            dim_score = max(0, min(5, dim_score))  # Clamp to 0-5
            breakdown[dim] = dim_score
            # Convert 0-5 to weighted contribution (max 30 total)
            total_score += dim_score * weight * 6  # 5 * 6 = 30 max

        breakdown["ai_notes"] = parsed.get("brief_notes", "")[:200]

        # Cache result
        cache_data = {"score": total_score, "breakdown": breakdown.copy()}
        self._cache_result(content_hash, cache_data)

        return min(total_score, 30.0), breakdown

    def _extract_content(self, data: Dict[str, Any], task_type: str) -> str:
        """Extract text content for evaluation based on task type."""
        content_parts = []

        if task_type == "blueprint":
            content_parts.extend([
                f"Title: {data.get('title', '')}",
                f"Genre: {data.get('genre', '')}",
                f"Setting: {data.get('setting', '')}",
                f"Core Conflict: {data.get('core_conflict', '')}",
                f"Themes: {', '.join(data.get('themes', []))}"
            ])

        elif task_type == "characters":
            characters = data.get("characters", [])
            for char in characters[:5]:  # Limit to 5 characters
                if isinstance(char, dict):
                    content_parts.append(
                        f"Character: {char.get('name', 'Unknown')}\n"
                        f"Background: {char.get('background', '')}\n"
                        f"Motivation: {char.get('motivation', '')}"
                    )

        elif task_type == "story_arc":
            chapters = data.get("chapters", [])
            for chap in chapters[:12]:  # Limit to 12 chapters
                if isinstance(chap, dict):
                    content_parts.append(
                        f"Chapter {chap.get('chapter', '?')}: {chap.get('title', '')}\n"
                        f"{chap.get('description', '')}"
                    )

        elif task_type == "chapters":
            chapters = data.get("chapters", [])
            for chap in chapters[:3]:  # Limit to 3 chapters for eval
                if isinstance(chap, dict):
                    content_parts.append(chap.get("content", ""))

        return "\n\n".join(content_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_evaluations),
            "api_errors": self.api_errors,
            "cache_size": len(self.cache)
        }


# Module-level evaluator instance
_evaluator: Optional[NarrativeEvaluator] = None


def get_evaluator(config_path: Optional[str] = None) -> NarrativeEvaluator:
    """Get or create the global evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = NarrativeEvaluator(config_path)
    return _evaluator


def calculate_narrative_score(
    data: Dict[str, Any],
    context: Dict[str, Any],
    task_type: str,
    config_path: Optional[str] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate narrative merit score using AI evaluation.

    This is the main entry point for the scoring module.

    Args:
        data: Parsed JSON data from miner response
        context: Context dict with user_input, blueprint, etc.
        task_type: Type of task ("blueprint", "characters", "story_arc", "chapters")
        config_path: Optional path to custom config file

    Returns:
        Tuple of (score 0-30, breakdown dict)
    """
    evaluator = get_evaluator(config_path)
    return evaluator.evaluate(data, context, task_type)


# Export functions
__all__ = [
    "calculate_narrative_score",
    "NarrativeEvaluator",
    "get_evaluator"
]
