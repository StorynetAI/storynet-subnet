"""
Narrative Merit Score Module
============================

Calculates Narrative Merit Score (30 points) using rule-based evaluation.

This module evaluates:
- Storytelling structure and flow
- Dialogue and emotional elements
- Vocabulary richness and diversity
- Internal consistency

All scoring is deterministic and requires no external dependencies.
"""

import json
import re
import hashlib
from typing import Dict, Any, Tuple, List
from collections import Counter


def calculate_narrative_score(
    data: Dict[str, Any],
    context: Dict[str, Any],
    task_type: str,
    config_path: str = None  # Kept for backward compatibility, not used
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate narrative merit score using rule-based evaluation.

    Args:
        data: Parsed JSON data from miner response
        context: Context dict with user_input, blueprint, etc.
        task_type: Type of task ("blueprint", "characters", "story_arc", "chapters")
        config_path: Unused, kept for backward compatibility

    Returns:
        Tuple of (score 0-30, breakdown dict)
    """
    breakdown = {
        "structure_flow": 0.0,
        "dialogue_emotion": 0.0,
        "vocabulary_richness": 0.0,
        "internal_consistency": 0.0,
        "evaluation_method": "rule_based"
    }

    # Extract content based on task type
    content = _extract_content(data, task_type)

    if not content or len(content) < 50:
        breakdown["evaluation_method"] = "insufficient_content"
        return 5.0, breakdown

    # 1. Structure & Flow (0-7.5 points)
    structure_score = _evaluate_structure_flow(content, data, task_type)
    breakdown["structure_flow"] = structure_score

    # 2. Dialogue & Emotional Elements (0-7.5 points)
    dialogue_score = _evaluate_dialogue_emotion(content)
    breakdown["dialogue_emotion"] = dialogue_score

    # 3. Vocabulary Richness (0-7.5 points)
    vocab_score = _evaluate_vocabulary_richness(content)
    breakdown["vocabulary_richness"] = vocab_score

    # 4. Internal Consistency (0-7.5 points)
    consistency_score = _evaluate_internal_consistency(data, task_type)
    breakdown["internal_consistency"] = consistency_score

    total_score = structure_score + dialogue_score + vocab_score + consistency_score
    return min(total_score, 30.0), breakdown


def _extract_content(data: Dict[str, Any], task_type: str) -> str:
    """Extract text content for evaluation based on task type."""
    content_parts = []

    if task_type == "blueprint":
        content_parts.extend([
            str(data.get("title", "")),
            str(data.get("genre", "")),
            str(data.get("setting", "")),
            str(data.get("core_conflict", "")),
            " ".join(data.get("themes", [])) if isinstance(data.get("themes"), list) else str(data.get("themes", ""))
        ])

    elif task_type == "characters":
        characters = data.get("characters", [])
        for char in characters[:5]:
            if isinstance(char, dict):
                content_parts.append(
                    f"{char.get('name', '')} {char.get('background', '')} {char.get('motivation', '')}"
                )

    elif task_type == "story_arc":
        chapters = data.get("chapters", [])
        for chap in chapters[:12]:
            if isinstance(chap, dict):
                content_parts.append(
                    f"{chap.get('title', '')} {chap.get('description', '')}"
                )

    elif task_type == "chapters":
        chapters = data.get("chapters", [])
        for chap in chapters[:3]:
            if isinstance(chap, dict):
                content_parts.append(str(chap.get("content", "")))

    return "\n".join(content_parts)


def _evaluate_structure_flow(content: str, data: Dict[str, Any], task_type: str) -> float:
    """
    Evaluate narrative structure and flow (0-7.5 points).

    Checks:
    - Paragraph/section organization
    - Sentence length variation
    - Transition words usage
    """
    score = 0.0

    # Check content length (longer = more developed)
    if len(content) > 200:
        score += 1.5
    if len(content) > 500:
        score += 1.0
    if len(content) > 1000:
        score += 0.5

    # Check for paragraph structure
    paragraphs = [p for p in content.split('\n') if p.strip()]
    if len(paragraphs) >= 3:
        score += 1.5
    elif len(paragraphs) >= 2:
        score += 0.75

    # Check sentence variation
    sentences = re.split(r'[。！？.!?]', content)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 3:
        lengths = [len(s) for s in sentences]
        if lengths:
            variance = sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)
            if variance > 100:  # Good variation
                score += 1.5
            elif variance > 50:
                score += 0.75

    # Check for transition words (Chinese and English)
    transition_words = [
        "然后", "接着", "随后", "因此", "所以", "但是", "然而", "不过",
        "首先", "其次", "最后", "同时", "此外", "另外", "总之",
        "then", "next", "however", "therefore", "finally", "meanwhile", "moreover"
    ]
    transition_count = sum(1 for word in transition_words if word in content)
    if transition_count >= 5:
        score += 1.5
    elif transition_count >= 2:
        score += 0.75

    return min(score, 7.5)


def _evaluate_dialogue_emotion(content: str) -> float:
    """
    Evaluate dialogue and emotional elements (0-7.5 points).

    Checks:
    - Presence of dialogue markers
    - Emotional vocabulary
    - Character interactions
    """
    score = 0.0

    # Check for dialogue markers
    dialogue_markers = ['"', '"', '"', '「', '」', '『', '』', '说道', '问道', '回答', 'said', 'asked', 'replied']
    dialogue_count = sum(content.count(marker) for marker in dialogue_markers)
    if dialogue_count >= 10:
        score += 2.5
    elif dialogue_count >= 5:
        score += 1.5
    elif dialogue_count >= 2:
        score += 0.75

    # Check for emotional vocabulary
    emotion_words = [
        # Chinese
        "喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "期待", "信任",
        "快乐", "痛苦", "紧张", "兴奋", "失望", "感动", "焦虑", "平静",
        "爱", "恨", "希望", "绝望", "勇敢", "害怕",
        # English
        "happy", "sad", "angry", "fear", "surprise", "love", "hate",
        "excited", "nervous", "hope", "despair", "brave", "scared"
    ]
    emotion_count = sum(1 for word in emotion_words if word in content)
    if emotion_count >= 8:
        score += 2.5
    elif emotion_count >= 4:
        score += 1.5
    elif emotion_count >= 2:
        score += 0.75

    # Check for action/interaction verbs
    action_words = [
        "走", "跑", "看", "听", "说", "想", "做", "拿", "打", "抓",
        "walk", "run", "look", "listen", "speak", "think", "make", "take"
    ]
    action_count = sum(content.count(word) for word in action_words)
    if action_count >= 15:
        score += 2.5
    elif action_count >= 8:
        score += 1.5
    elif action_count >= 3:
        score += 0.75

    return min(score, 7.5)


def _evaluate_vocabulary_richness(content: str) -> float:
    """
    Evaluate vocabulary richness and diversity (0-7.5 points).

    Checks:
    - Unique word ratio
    - Character/word diversity
    - Absence of excessive repetition
    """
    score = 0.0

    if len(content) < 20:
        return 0.0

    # For Chinese: character-level analysis
    # For English: word-level analysis

    # Character diversity (works for both Chinese and English)
    chars = [c for c in content if c.strip() and not c.isspace()]
    if chars:
        unique_chars = set(chars)
        char_diversity = len(unique_chars) / len(chars)

        if char_diversity > 0.4:
            score += 2.5
        elif char_diversity > 0.3:
            score += 1.5
        elif char_diversity > 0.2:
            score += 0.75

    # Word-level analysis (split by spaces and punctuation)
    words = re.findall(r'[\w\u4e00-\u9fff]+', content)
    if len(words) >= 10:
        unique_words = set(words)
        word_diversity = len(unique_words) / len(words)

        if word_diversity > 0.6:
            score += 2.5
        elif word_diversity > 0.4:
            score += 1.5
        elif word_diversity > 0.3:
            score += 0.75

    # Check for excessive repetition (penalize if same phrase repeated too much)
    # Find 3-grams and check repetition
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        max_repeat = max(trigram_counts.values()) if trigram_counts else 0

        if max_repeat <= 2:
            score += 2.5
        elif max_repeat <= 4:
            score += 1.5
        elif max_repeat <= 6:
            score += 0.75

    return min(score, 7.5)


def _evaluate_internal_consistency(data: Dict[str, Any], task_type: str) -> float:
    """
    Evaluate internal consistency (0-7.5 points).

    Checks:
    - Named entity consistency
    - Theme/setting consistency
    - Structural completeness
    """
    score = 0.0

    if task_type == "blueprint":
        # Check all required fields are present and non-empty
        required = ["title", "genre", "setting", "core_conflict", "themes"]
        filled = sum(1 for f in required if data.get(f))
        score += (filled / len(required)) * 5.0

        # Check themes is a list with items
        themes = data.get("themes", [])
        if isinstance(themes, list) and len(themes) >= 2:
            score += 2.5
        elif isinstance(themes, list) and len(themes) >= 1:
            score += 1.25

    elif task_type == "characters":
        characters = data.get("characters", [])
        if isinstance(characters, list):
            # Check character count
            if len(characters) >= 5:
                score += 2.5
            elif len(characters) >= 3:
                score += 1.5

            # Check each character has required fields
            required_char_fields = ["name", "background", "motivation"]
            complete_chars = 0
            for char in characters[:5]:
                if isinstance(char, dict):
                    if all(char.get(f) for f in required_char_fields):
                        complete_chars += 1

            score += (complete_chars / 5) * 5.0

    elif task_type == "story_arc":
        chapters = data.get("chapters", [])
        if isinstance(chapters, list):
            # Check chapter count
            if len(chapters) >= 12:
                score += 2.5
            elif len(chapters) >= 8:
                score += 1.5
            elif len(chapters) >= 5:
                score += 0.75

            # Check each chapter has required fields
            required_chap_fields = ["title", "description"]
            complete_chaps = 0
            for chap in chapters[:12]:
                if isinstance(chap, dict):
                    if all(chap.get(f) for f in required_chap_fields):
                        complete_chaps += 1

            if chapters:
                score += (complete_chaps / min(len(chapters), 12)) * 5.0

    elif task_type == "chapters":
        chapters = data.get("chapters", [])
        if isinstance(chapters, list) and chapters:
            # Check content length and quality
            total_content_len = 0
            has_choices = 0

            for chap in chapters:
                if isinstance(chap, dict):
                    content = chap.get("content", "")
                    total_content_len += len(content)
                    if chap.get("choices"):
                        has_choices += 1

            # Content length score
            if total_content_len > 2000:
                score += 3.75
            elif total_content_len > 1000:
                score += 2.5
            elif total_content_len > 500:
                score += 1.25

            # Choices present score
            if has_choices >= len(chapters):
                score += 3.75
            elif has_choices > 0:
                score += 1.875

    return min(score, 7.5)


# Export
__all__ = ["calculate_narrative_score"]
