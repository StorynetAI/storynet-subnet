"""
StoryNet Bittensor Subnet Protocol Definition
=============================================

This module defines the Synapse communication protocol between Miners and Validators.

Synapse = Request/Response format for AI generation tasks.

Task Types:
- blueprint: Generate story world and themes
- characters: Generate 5 character profiles
- story_arc: Generate 12-chapter story structure
- chapters: Generate detailed chapter content with choices

PROTOCOL VERSION: 3.2.1
CHANGES FROM 3.2.0:
- Fixed characters field type: Dict[str, Any] -> List[Dict[str, Any]]
- This aligns with actual usage in scoring/structure.py
- Fixes Pydantic validation error when passing character list

CHANGES FROM 3.1.0:
- Added model_info field for transparency and quality control
- Miners must disclose which model they use for generation
- Enables validators to apply dynamic reward multipliers based on model quality

CHANGES FROM 3.0.0:
- Removed custom get_total_size() override
- Allow Bittensor framework to handle size calculation and data transmission
- Rely on Bittensor's built-in dual-channel mechanism for large data

CHANGES FROM 2.1.0:
- Removed compression (caused double-encoding overhead)
- Use simple types directly (str, Dict, List, Optional)
- Let Bittensor handle serialization (standard practice)
- Added required_hash_fields for integrity verification
- Simplified field structure following OCR/text-prompting patterns
- Removed redundant helper methods

DESIGN PHILOSOPHY:
- Keep It Simple: Use basic Python types
- Trust Bittensor: Let framework handle serialization and transmission
- Follow Standards: Based on OCR subnet and text-prompting subnet patterns
- Avoid Over-Engineering: No manual size calculations or encoding

REFERENCES:
- OCR Subnet: https://github.com/opentensor/ocr_subnet
- Text-Prompting: https://github.com/opentensor/text-prompting
- Official Template: https://github.com/opentensor/bittensor-subnet-template
"""

import bittensor as bt
from typing import Optional, Dict, Any, List
from pydantic import Field, field_validator
import json
import sys


class StoryGenerationSynapse(bt.Synapse):
    """
    Synapse for story generation tasks (Protocol v3.0.0).

    This is sent from Validator to Miner with task requirements,
    and Miner fills in the response fields.

    DESIGN: Simple, direct field types. Bittensor handles serialization automatically.

    Request Flow:
        Validator creates Synapse → Bittensor serializes to HTTP headers →
        Miner receives → Miner processes → Miner fills response fields →
        Bittensor serializes response → Validator receives result

    Attributes:
        protocol_version: Protocol version for compatibility checking (default: "3.0.0")
        task_type: Type of generation task (required, filled by Validator)

        # Request fields (filled by Validator)
        user_input: User's story request
        blueprint: Story blueprint (for characters/story_arc/chapters tasks)
        characters: Character profiles (for story_arc/chapters tasks)
        story_arc: Story structure (for chapters task)
        chapter_ids: Chapter IDs to generate (for chapters task)

        # Response fields (filled by Miner)
        output_data: Generated content as Dict
        generation_time: Time taken to generate in seconds
        miner_version: Miner software version

        # Metadata (auto-filled by framework)
        miner_hotkey: Miner's hotkey address
        validator_hotkey: Validator's hotkey address

        # Integrity verification
        required_hash_fields: Fields to include in body hash
    """

    # Protocol version
    protocol_version: str = Field(
        default="3.2.1",
        description="Protocol version for compatibility checking"
    )

    # Request fields (filled by Validator) - required
    # NOTE: Must use default="" instead of ... for Bittensor serialization to work
    task_type: str = Field(
        default="",
        description="Task type: 'blueprint' | 'characters' | 'story_arc' | 'chapters'"
    )

    user_input: str = Field(
        default="",
        description="User's story request (e.g., '一个关于AI觉醒的科幻故事')"
    )

    # Request fields (filled by Validator) - optional, task-specific
    blueprint: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Story blueprint (required for characters/story_arc/chapters tasks)"
    )

    characters: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Character profiles (required for story_arc/chapters tasks)"
    )

    story_arc: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Story structure (required for chapters task)"
    )

    chapter_ids: Optional[List[int]] = Field(
        default=None,
        description="Chapter IDs to generate (e.g., [1, 2, 3]) (required for chapters task)"
    )

    # Response fields (filled by Miner)
    output_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generated content (filled by Miner)"
    )

    generation_time: float = Field(
        default=0.0,
        description="Time taken to generate in seconds"
    )

    miner_version: str = Field(
        default="",
        description="Miner software version"
    )

    model_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model information for transparency and quality scoring"
    )

    # Metadata (auto-filled by framework)
    miner_hotkey: Optional[str] = Field(
        default=None,
        description="Miner's hotkey address"
    )

    validator_hotkey: Optional[str] = Field(
        default=None,
        description="Validator's hotkey address"
    )

    # Validation
    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task_type is one of the allowed values."""
        # Allow empty string for initialization (Bittensor creates empty Synapse first)
        if v == "":
            return v

        allowed = ["blueprint", "characters", "story_arc", "chapters"]
        if v not in allowed:
            raise ValueError(f"task_type must be one of {allowed}, got {v}")
        return v

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, v: str) -> str:
        """Validate protocol_version format (X.Y.Z)."""
        # Allow empty string for initialization (Bittensor creates empty Synapse first)
        if v == "":
            return v

        if not v or len(v.split(".")) != 3:
            raise ValueError(f"protocol_version must be in format X.Y.Z, got {v}")
        return v

    def validate_input_fields(self) -> bool:
        """
        Validate that required fields are present for the given task_type.

        Returns:
            True if all required fields are present, False otherwise.
        """
        if self.task_type == "blueprint":
            # Only user_input required
            return True

        elif self.task_type == "characters":
            # blueprint and user_input required
            return self.blueprint is not None

        elif self.task_type == "story_arc":
            # blueprint, characters, and user_input required
            return (
                self.blueprint is not None and
                self.characters is not None
            )

        elif self.task_type == "chapters":
            # All fields required
            return (
                self.blueprint is not None and
                self.characters is not None and
                self.story_arc is not None and
                self.chapter_ids is not None
            )

        return False

    def get_required_output_fields(self) -> List[str]:
        """
        Get list of required fields in output_data for this task_type.

        Returns:
            List of field names that must exist in the output data.
        """
        if self.task_type == "blueprint":
            return [
                "title",
                "genre",
                "setting",
                "core_conflict",
                "themes",
                "tone",
                "target_audience"
            ]

        elif self.task_type == "characters":
            return ["characters"]  # Must be array of 5 character objects

        elif self.task_type == "story_arc":
            return [
                "title",
                "description",
                "chapters",
                "arcs",
                "themes",
                "hooks"
            ]

        elif self.task_type == "chapters":
            return ["chapters"]  # Must be array of chapter objects

        return []

    def deserialize(self) -> Optional[Dict[str, Any]]:
        """
        Deserialize the Miner response.

        Returns:
            The output_data dictionary, or None if not set.

        Note: This follows the pattern from OCR subnet and text-prompting subnet.
        """
        return self.output_data

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"StoryGenerationSynapse("
            f"v{self.protocol_version}, "
            f"task={self.task_type}, "
            f"user_input='{self.user_input[:30]}...', "
            f"has_output={self.output_data is not None}, "
            f"gen_time={self.generation_time:.2f}s)"
        )


# Helper functions for creating Synapses

def create_blueprint_synapse(user_input: str) -> StoryGenerationSynapse:
    """
    Create a Synapse for blueprint generation.

    Args:
        user_input: User's story request (e.g., "一个关于赛博朋克黑客的故事")

    Returns:
        StoryGenerationSynapse configured for blueprint task

    Example:
        >>> synapse = create_blueprint_synapse("一个关于太空探险的故事")
        >>> # Send to miner via dendrite.query()
    """
    return StoryGenerationSynapse(
        task_type="blueprint",
        user_input=user_input
    )


def create_characters_synapse(
    blueprint: Dict[str, Any],
    user_input: str
) -> StoryGenerationSynapse:
    """
    Create a Synapse for characters generation.

    Args:
        blueprint: Output from blueprint task
        user_input: Original user input

    Returns:
        StoryGenerationSynapse configured for characters task
    """
    return StoryGenerationSynapse(
        task_type="characters",
        user_input=user_input,
        blueprint=blueprint
    )


def create_story_arc_synapse(
    blueprint: Dict[str, Any],
    characters: List[Dict[str, Any]],
    user_input: str
) -> StoryGenerationSynapse:
    """
    Create a Synapse for story arc generation.

    Args:
        blueprint: Output from blueprint task
        characters: Output from characters task
        user_input: Original user input

    Returns:
        StoryGenerationSynapse configured for story_arc task
    """
    return StoryGenerationSynapse(
        task_type="story_arc",
        user_input=user_input,
        blueprint=blueprint,
        characters=characters
    )


def create_chapters_synapse(
    blueprint: Dict[str, Any],
    characters: List[Dict[str, Any]],
    story_arc: Dict[str, Any],
    chapter_ids: List[int],
    user_input: str
) -> StoryGenerationSynapse:
    """
    Create a Synapse for chapters generation.

    Args:
        blueprint: Output from blueprint task
        characters: Output from characters task
        story_arc: Output from story_arc task
        chapter_ids: List of chapter IDs to generate (e.g., [1, 2, 3])
        user_input: Original user input

    Returns:
        StoryGenerationSynapse configured for chapters task
    """
    return StoryGenerationSynapse(
        task_type="chapters",
        user_input=user_input,
        blueprint=blueprint,
        characters=characters,
        story_arc=story_arc,
        chapter_ids=chapter_ids
    )


# Export all public classes and functions
__all__ = [
    "StoryGenerationSynapse",
    "create_blueprint_synapse",
    "create_characters_synapse",
    "create_story_arc_synapse",
    "create_chapters_synapse"
]
