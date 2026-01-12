"""
StoryNet Bittensor Validator
============================

This validator:
1. Generates tasks for miners
2. Queries miners with different task types
3. Evaluates responses using scoring system
4. Calculates and sets weights on-chain
5. Implements anti-cheating mechanisms

Usage:
    python neurons/validator.py \
        --netuid 92 \
        --wallet.name my_validator \
        --wallet.hotkey default \
        --logging.info
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import bittensor as bt
import torch
import yaml
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.protocol import (
    StoryGenerationSynapse,
    create_blueprint_synapse,
    create_characters_synapse,
    create_story_arc_synapse,
    create_chapters_synapse
)
from template.utils import (
    Timer,
    exponential_moving_average,
    normalize_weights
)
from scoring import (
    calculate_technical_score,
    calculate_structure_score,
    calculate_content_score,
    calculate_narrative_score
)

# Load environment variables
load_dotenv()


class StoryValidator:
    """
    StoryNet Validator that evaluates miners and distributes rewards.

    The validator:
    1. Periodically queries miners with story generation tasks
    2. Scores responses using 3-part system (Technical + Structure + Content)
    3. Updates EMA scores and calculates weights
    4. Sets weights on-chain every N queries
    5. Detects and blacklists cheating miners
    """

    def __init__(self, config: bt.Config):
        """Initialize the validator."""
        self.config = config
        bt.logging.info("Initializing StoryNet Validator...")

        # Initialize Bittensor components
        self.wallet = bt.Wallet(config=self.config)
        self.subtensor = bt.Subtensor(config=self.config)
        self.metagraph = bt.Metagraph(netuid=self.config.netuid, network=self.subtensor.network)
        self.dendrite = bt.Dendrite(wallet=self.wallet)

        # Configuration
        self.query_interval = int(os.getenv("VALIDATOR_QUERY_INTERVAL", "12"))
        self.timeout = int(os.getenv("VALIDATOR_TIMEOUT", "60"))
        self.ema_alpha = float(os.getenv("EMA_ALPHA", "0.1"))
        self.temperature = float(os.getenv("SOFTMAX_TEMPERATURE", "2.0"))
        self.weight_update_frequency = int(os.getenv("WEIGHT_UPDATE_FREQ", "100"))

        # Weight submission tracking (block-based rate limiting)
        self.last_weights_block = 0
        self.weights_rate_limit = 100  # Default, will be updated from chain

        # Task distribution (blueprint:40%, characters:25%, story_arc:25%, chapters:10%)
        self.task_distribution = {
            "blueprint": 0.40,
            "characters": 0.25,
            "story_arc": 0.25,
            "chapters": 0.10
        }

        # State
        self.scores = {}  # {miner_uid: ema_score}
        self.history = deque(maxlen=1000)  # Historical responses for plagiarism detection
        self.blacklist = set()  # Blacklisted miner UIDs
        self.violations = {}  # {miner_uid: violation_count}

        # Statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.total_rewards = 0.0

        # Sample data for task generation
        self.sample_prompts = [
            "一个关于赛博朋克黑客的故事",
            "一个关于太空探险的故事",
            "一个古代武侠传奇故事",
            "一个末日生存的故事",
            "一个都市悬疑推理故事",
            "一个奇幻魔法世界的故事",
            "一个时间旅行的故事",
            "一个AI觉醒的故事"
        ]

        # Load model quality policy (Protocol v3.2.0)
        self.model_policy = self._load_model_policy()

        bt.logging.info(f"✅ Wallet: {self.wallet.hotkey.ss58_address}")
        bt.logging.info(f"✅ Netuid: {self.config.netuid}")
        bt.logging.info(f"✅ Query interval: {self.query_interval}s")
        bt.logging.info(f"✅ Model quality policy loaded")

    def _load_model_policy(self) -> Dict[str, Any]:
        """Load model quality policy from YAML file."""
        policy_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "validators/config/model_policy.yaml"
        )

        try:
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy = yaml.safe_load(f)
                return policy
        except Exception as e:
            bt.logging.warning(f"Failed to load model policy: {e}, using defaults")
            # Return default policy if file not found
            return {
                "quality_policy": {
                    "min_quality_score": 0.6,
                    "mode_multipliers": {
                        "local": 1.0,
                        "vllm": 1.0,
                        "custom": 1.0,
                        "api": 1.0,
                        "unknown": 1.0
                    },
                    "recommended_models": [],
                    "blacklisted_models": [],
                    "penalties": {
                        "no_model_info": 1.0
                    }
                }
            }

    def apply_model_quality_multiplier(
        self,
        base_score: float,
        model_info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply model quality policy to adjust score (Protocol v3.2.0).

        Args:
            base_score: Base score from content/structure/technical evaluation (0-100)
            model_info: Model information from miner

        Returns:
            Tuple of (adjusted_score, multiplier_breakdown)
        """
        policy = self.model_policy.get("quality_policy", {})
        multiplier_info = {
            "mode_multiplier": 1.0,
            "model_bonus": 1.0,
            "penalty": 1.0,
            "final_multiplier": 1.0
        }

        # Check if model_info is missing or unknown
        if not model_info or model_info.get("mode") == "unknown":
            penalty = policy.get("penalties", {}).get("no_model_info", 0.5)
            multiplier_info["penalty"] = penalty
            multiplier_info["final_multiplier"] = penalty
            final_score = base_score * penalty
            bt.logging.warning(f"⚠️  No model info provided, applying {penalty}x penalty")
            return final_score, multiplier_info

        mode = model_info.get("mode", "unknown")
        model_name = model_info.get("name", "unknown")

        # 1. Apply mode multiplier
        mode_multipliers = policy.get("mode_multipliers", {})
        mode_mult = mode_multipliers.get(mode, 1.0)
        multiplier_info["mode_multiplier"] = mode_mult

        # 2. Check blacklist (instant disqualification)
        blacklist = policy.get("blacklisted_models", [])
        if any(blacklisted in model_name for blacklisted in blacklist):
            bt.logging.error(f"🚫 Blacklisted model detected: {model_name}")
            return 0.0, multiplier_info

        # 3. Check recommended models for bonus
        model_bonus = 1.0
        for rec_model in policy.get("recommended_models", []):
            if rec_model["name"] in model_name:
                model_bonus = rec_model.get("bonus", 1.0)
                multiplier_info["model_bonus"] = model_bonus
                bt.logging.info(f"✨ Recommended model {model_name}, {model_bonus}x bonus")
                break

        # Calculate final multiplier
        final_multiplier = mode_mult * model_bonus
        multiplier_info["final_multiplier"] = final_multiplier

        # Apply multiplier
        final_score = base_score * final_multiplier

        # Apply minimum quality threshold
        min_quality = policy.get("min_quality_score", 0.6)
        normalized_score = base_score / 100.0
        if normalized_score < min_quality:
            bt.logging.warning(
                f"⚠️  Score {base_score:.2f} below minimum quality {min_quality*100:.2f}, "
                f"setting to 0"
            )
            return 0.0, multiplier_info

        return final_score, multiplier_info

    def select_task_type(self, block: int) -> str:
        """
        Deterministically select a task type based on block number.

        All validators at the same block will select the same task type,
        ensuring consensus across the network.

        Args:
            block: Current block number used as random seed

        Returns:
            Selected task type string
        """
        # Save current random state
        state = random.getstate()

        # Use block as seed for deterministic selection
        random.seed(block)
        task_type = random.choices(
            list(self.task_distribution.keys()),
            weights=list(self.task_distribution.values())
        )[0]

        # Restore random state to not affect other random operations
        random.setstate(state)

        return task_type

    def create_task(self, task_type: str, block: int) -> Tuple[StoryGenerationSynapse, Dict[str, Any]]:
        """
        Create a task synapse with mock context using deterministic selection.

        All validators at the same block will create the same task,
        ensuring consensus across the network.

        Args:
            task_type: Type of task to create
            block: Current block number used as random seed

        Returns:
            Tuple of (synapse, context)
        """
        # Save current random state
        state = random.getstate()

        # Use block + offset as seed (offset to get different value than task_type selection)
        random.seed(block + 1000)
        user_input = random.choice(self.sample_prompts)

        # Restore random state
        random.setstate(state)
        context = {"user_input": user_input}

        if task_type == "blueprint":
            synapse = create_blueprint_synapse(user_input)

        elif task_type == "characters":
            # Mock blueprint for characters task
            mock_blueprint = {
                "title": "Sample Story",
                "genre": "Sci-Fi",
                "setting": "Future World",
                "core_conflict": "Human vs AI",
                "themes": ["Technology", "Ethics"],
                "tone": "Suspenseful",
                "target_audience": "Adults"
            }
            synapse = create_characters_synapse(mock_blueprint, user_input)
            context["blueprint"] = mock_blueprint

        elif task_type == "story_arc":
            # Mock blueprint and characters
            mock_blueprint = {
                "title": "Sample Story",
                "genre": "Sci-Fi",
                "setting": "Future World",
                "core_conflict": "Human vs AI",
                "themes": ["Technology", "Ethics"],
                "tone": "Suspenseful",
                "target_audience": "Adults"
            }
            mock_characters = [
                {"id": "protagonist", "name": "Hero", "archetype": "Hero"},
                {"id": "ally", "name": "Ally", "archetype": "Helper"},
                {"id": "rival", "name": "Rival", "archetype": "Villain"},
                {"id": "mentor", "name": "Mentor", "archetype": "Sage"},
                {"id": "wildcard", "name": "Wildcard", "archetype": "Mystery"}
            ]
            synapse = create_story_arc_synapse(mock_blueprint, mock_characters, user_input)
            context["blueprint"] = mock_blueprint
            context["characters"] = mock_characters

        elif task_type == "chapters":
            # Mock complete context
            mock_blueprint = {"title": "Sample Story"}
            mock_characters = []
            mock_story_arc = {
                "title": "Sample Story",
                "chapters": [{"id": i} for i in range(1, 13)]
            }
            chapter_ids = [1]
            synapse = create_chapters_synapse(
                mock_blueprint, mock_characters, mock_story_arc,
                chapter_ids, user_input
            )
            context["blueprint"] = mock_blueprint
            context["characters"] = mock_characters
            context["story_arc"] = mock_story_arc

        return synapse, context

    async def query_miners(
        self,
        synapse: StoryGenerationSynapse,
        miners: List[bt.AxonInfo]
    ) -> List[StoryGenerationSynapse]:
        """
        Query multiple miners with a task.

        Args:
            synapse: Task synapse
            miners: List of miner axons

        Returns:
            List of responses
        """
        try:
            responses = await self.dendrite.forward(
                axons=miners,
                synapse=synapse,
                timeout=self.timeout
            )
            return responses
        except Exception as e:
            bt.logging.error(f"Error querying miners: {e}")
            return [synapse] * len(miners)  # Return empty responses on error

    def score_response(
        self,
        response: StoryGenerationSynapse,
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score a miner's response using 3-part scoring system with model quality multiplier (Protocol v3.2.0).

        Scoring components:
        1. Technical Score (30 points): Validity, performance, format
        2. Structure Score (40 points): Completeness, field quality
        3. Content Score (30 points): Creativity, coherence, relevance
        4. Model Quality Multiplier: Based on model mode, name, and quality policy

        Args:
            response: Miner's response synapse
            context: Task context

        Returns:
            Tuple of (final_score, breakdown) where final_score includes model quality multiplier
        """
        # Initialize breakdown
        breakdown = {
            "technical": 0.0,
            "structure": 0.0,
            "content": 0.0,
            "total": 0.0
        }

        # Get output data (v3.0.0: direct Dict access)
        data = response.output_data
        if not data:
            return 0.0, breakdown

        # Get required fields
        required_fields = response.get_required_output_fields()

        # 1. Technical Score (20 points) - Objective, rule-based
        # v3.0.0: convert output_data to JSON string for technical scoring
        output_json_str = json.dumps(data, ensure_ascii=False)
        tech_score_raw, tech_breakdown = calculate_technical_score(
            output_json_str,
            response.generation_time,
            response.task_type,
            required_fields
        )
        # Scale from 30 to 20 points
        tech_score = tech_score_raw * (20.0 / 30.0)
        breakdown["technical"] = tech_score
        breakdown["technical_breakdown"] = tech_breakdown

        # 2. Structure Score (30 points) - Objective, rule-based
        struct_score_raw, struct_breakdown = calculate_structure_score(
            data,
            response.task_type
        )
        # Scale from 40 to 30 points
        struct_score = struct_score_raw * (30.0 / 40.0)
        breakdown["structure"] = struct_score
        breakdown["structure_breakdown"] = struct_breakdown

        # 3. Content Score (20 points) - Semi-objective, heuristic-based
        content_score_raw, content_breakdown = calculate_content_score(
            data,
            context,
            response.task_type,
            history=[h["output_data"] for h in list(self.history)[-20:]],
            use_embeddings=False  # Set to True when OpenAI embeddings are available
        )
        # Scale from 30 to 20 points
        content_score = content_score_raw * (20.0 / 30.0)
        breakdown["content"] = content_score
        breakdown["content_breakdown"] = content_breakdown

        # 4. Narrative Merit Score (30 points) - AI-based evaluation
        # This uses server-side LLM evaluation with private prompts
        narrative_score, narrative_breakdown = calculate_narrative_score(
            data,
            context,
            response.task_type
        )
        breakdown["narrative"] = narrative_score
        breakdown["narrative_breakdown"] = narrative_breakdown

        # Base score (0-100)
        base_score = tech_score + struct_score + content_score + narrative_score
        breakdown["base_score"] = base_score

        # Apply model quality multiplier (Protocol v3.2.0)
        model_info = response.model_info if hasattr(response, 'model_info') else {}
        final_score, multiplier_breakdown = self.apply_model_quality_multiplier(
            base_score,
            model_info
        )

        # Add multiplier info to breakdown
        breakdown["model_quality"] = multiplier_breakdown
        breakdown["total"] = final_score

        return final_score, breakdown

    def detect_plagiarism(
        self,
        response: StoryGenerationSynapse,
        all_responses: List[StoryGenerationSynapse]
    ) -> Tuple[bool, str, float]:
        """
        Detect plagiarism in response (Protocol v3.0.0).

        Args:
            response: Current response to check
            all_responses: All responses in current batch

        Returns:
            Tuple of (is_plagiarism, type, similarity)
        """
        # First check if response is None or doesn't have output_data
        if response is None or not hasattr(response, 'output_data'):
            return False, "", 0.0

        current_data = response.output_data
        if not current_data:
            return False, "", 0.0

        # Check against current batch (cross-miner copying)
        for other in all_responses:
            if other == response:
                continue

            # Skip if other response is None or invalid
            if other is None or not hasattr(other, 'output_data'):
                continue

            other_data = other.output_data
            if not other_data:
                continue

            similarity = self.calculate_similarity(current_data, other_data)

            if similarity > 0.95:
                return True, "cross_miner_copy", similarity

        # Check against history (template reuse)
        for historical in list(self.history)[-50:]:
            historical_data = historical.get("output_data")
            if not historical_data:
                continue

            similarity = self.calculate_similarity(current_data, historical_data)

            if similarity > 0.90:
                return True, "template_reuse", similarity

        return False, "original", 0.0

    def calculate_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate simple similarity between two data objects."""
        str1 = json.dumps(data1, ensure_ascii=False, sort_keys=True)
        str2 = json.dumps(data2, ensure_ascii=False, sort_keys=True)

        # Character-level bigrams
        def get_bigrams(s: str) -> set:
            return set(s[i:i+2] for i in range(len(s) - 1))

        bigrams1 = get_bigrams(str1)
        bigrams2 = get_bigrams(str2)

        if not bigrams1 or not bigrams2:
            return 0.0

        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)

        return intersection / union if union > 0 else 0.0

    def _is_miner_available(self, uid: int, axon: bt.AxonInfo) -> bool:
        """
        Check if a miner is available for querying.

        This follows the official Bittensor subnet template pattern.

        Filters out:
        - 0.0.0.0 IP addresses (unregistered/invalid)
        - Invalid ports
        - Missing hotkeys

        Args:
            uid: The UID to check
            axon: AxonInfo object to check

        Returns:
            True if miner is available, False otherwise
        """
        # Check if IP is not None or empty
        if not axon.ip:
            return False

        # Check if IP is not 0.0.0.0
        if axon.ip == "0.0.0.0":
            bt.logging.debug(f"Filtered axon with 0.0.0.0 IP (hotkey: {axon.hotkey[:8]}...)")
            return False

        # Check if port is valid (not 0 and in valid range)
        if axon.port <= 0 or axon.port > 65535:
            bt.logging.debug(f"Filtered axon with invalid port {axon.port}")
            return False

        # Check if hotkey is not empty
        if not axon.hotkey:
            bt.logging.debug("Filtered axon with missing hotkey")
            return False

        return True

    def update_ema_scores(self, new_scores: Dict[int, float]):
        """Update EMA scores for miners."""
        for uid, score in new_scores.items():
            if uid not in self.scores:
                self.scores[uid] = score
            else:
                self.scores[uid] = exponential_moving_average(
                    score,
                    self.scores[uid],
                    self.ema_alpha
                )

    def get_burn_uid(self) -> int:
        """
        Get the subnet owner's UID (burn_uid).
        The owner UID is excluded from miner weight calculation and receives remaining weight.
        This is inspired by DogeLayer's burn_uid mechanism.

        Returns:
            The UID of the subnet owner, or -1 if not found.
        """
        try:
            # Get subnet owner's hotkey from chain
            # Note: API is get_subnet_owner_hotkey() not get_subnet_owner()
            owner_hotkey = self.subtensor.get_subnet_owner_hotkey(self.config.netuid)

            if owner_hotkey is None:
                bt.logging.warning(f"Could not get subnet owner for netuid {self.config.netuid}")
                return -1

            # Find the UID corresponding to this hotkey in metagraph
            for uid in range(len(self.metagraph.hotkeys)):
                if self.metagraph.hotkeys[uid] == owner_hotkey:
                    bt.logging.info(f"🔥 Burn UID (subnet owner): {uid}")
                    return uid

            bt.logging.warning(f"Subnet owner hotkey not found in metagraph: {owner_hotkey}")
            return -1

        except Exception as e:
            bt.logging.error(f"Error getting burn_uid: {e}")
            return -1

    def calculate_weights(self) -> Dict[int, float]:
        """
        Calculate weights using 3-factor system (inspired by SoulX):
        - 15% Stake weight (防止新矿工作弊)
        - 75% Quality score (当前表现)
        - 10% Historical score (长期稳定性)

        Burn UID mechanism (inspired by DogeLayer):
        - Owner UID is excluded from miner scoring
        - Remaining weight (after miners) is assigned to Owner
        """
        if not self.scores:
            return {}

        # Get burn_uid (subnet owner) to exclude from miner weights
        burn_uid = self.get_burn_uid()

        # Calculate composite scores with stake consideration
        # IMPORTANT: Exclude burn_uid (owner) from miner scoring
        composite_scores = {}

        for uid, quality_score in self.scores.items():
            # Skip burn_uid - owner is not a miner
            if uid == burn_uid:
                bt.logging.debug(f"Skipping burn_uid {uid} from miner scoring")
                continue

            # 1. Get stake weight
            try:
                stake = self.metagraph.S[uid].item()
                max_stake = max(self.metagraph.S).item() if len(self.metagraph.S) > 0 else 1.0
                stake_weight = stake / max_stake if max_stake > 0 else 0.0
            except Exception as e:
                bt.logging.warning(f"Failed to get stake for UID {uid}: {e}")
                stake_weight = 0.0

            # 2. Get historical score (从历史记录计算平均)
            historical_scores = [
                h["score"] for h in list(self.history)[-50:]
                if h.get("uid") == uid
            ]
            historical_avg = (
                sum(historical_scores) / len(historical_scores)
                if historical_scores else quality_score
            )
            historical_score = historical_avg / 100.0  # Normalize to 0-1

            # 3. Normalize quality score to 0-1
            normalized_quality = quality_score / 100.0

            # 4. Composite score (15% stake + 75% quality + 10% history)
            composite = (
                0.15 * stake_weight +
                0.75 * normalized_quality +
                0.10 * historical_score
            )

            composite_scores[uid] = composite

        # Apply temperature (增加竞争差异)
        incentives = {
            uid: score ** self.temperature
            for uid, score in composite_scores.items()
        }

        # Normalize miner weights to sum to 1.0 first
        weights = normalize_weights(incentives)

        # Apply minimum weight (防止完全归零)
        min_weight = 0.001
        weights = {uid: max(w, min_weight) for uid, w in weights.items()}

        # Re-normalize miner weights
        weights = normalize_weights(weights)

        # === Burn UID mechanism ===
        # Scale miner weights to leave room for owner, then assign remaining to owner
        # Owner gets a portion based on miner count (more miners = less owner share)
        # This ensures owner participates in emissions while not competing with miners
        if burn_uid >= 0:
            num_miners = len(weights)
            if num_miners > 0:
                # Owner share: starts at 50% with few miners, decreases as more miners join
                # Formula: owner_share = max(0.1, 0.5 - 0.01 * num_miners)
                # With 10 miners: 40%, 20 miners: 30%, 40 miners: 10% (minimum)
                owner_share = max(0.10, 0.50 - 0.01 * num_miners)
                miner_share = 1.0 - owner_share

                # Scale miner weights
                weights = {uid: w * miner_share for uid, w in weights.items()}

                # Assign remaining weight to owner
                weights[burn_uid] = owner_share

                bt.logging.info(
                    f"🔥 Burn UID allocation: owner_uid={burn_uid}, "
                    f"owner_share={owner_share:.2%}, miner_share={miner_share:.2%} "
                    f"({num_miners} miners)"
                )
            else:
                # No miners, give all weight to owner
                weights[burn_uid] = 1.0
                bt.logging.info(f"🔥 No miners found, all weight to owner UID {burn_uid}")

        return weights

    def can_set_weights(self) -> bool:
        """
        Check if we can set weights based on chain rate limit.

        Returns:
            True if enough blocks have passed since last weight submission
        """
        try:
            current_block = self.subtensor.block
            blocks_since_last = current_block - self.last_weights_block

            # Get rate limit from chain hyperparameters
            try:
                self.weights_rate_limit = self.subtensor.weights_rate_limit(self.config.netuid)
            except Exception:
                self.weights_rate_limit = 100  # Fallback default

            can_set = blocks_since_last >= self.weights_rate_limit

            if not can_set:
                blocks_remaining = self.weights_rate_limit - blocks_since_last
                bt.logging.debug(
                    f"⏳ Rate limit: {blocks_remaining} blocks remaining "
                    f"(current: {current_block}, last: {self.last_weights_block}, limit: {self.weights_rate_limit})"
                )
            else:
                bt.logging.info(
                    f"✅ Can set weights: {blocks_since_last} blocks since last submission "
                    f"(limit: {self.weights_rate_limit})"
                )

            return can_set

        except Exception as e:
            bt.logging.warning(f"Error checking weight rate limit: {e}")
            return True  # Allow on error to not block weight setting

    async def set_weights(self):
        """Set weights on blockchain with rate limit check."""
        try:
            # Check rate limit first
            if not self.can_set_weights():
                bt.logging.info("⏳ Skipping weight submission (rate limit)")
                return

            weights_dict = self.calculate_weights()

            if not weights_dict:
                bt.logging.warning("No weights to set")
                return

            # Log weight calculation details
            bt.logging.info("Weight calculation details:")
            sorted_weights = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
            for uid, weight in sorted_weights[:10]:  # Show top 10
                # Get miner info
                try:
                    axon = self.metagraph.axons[uid]
                    stake = self.metagraph.S[uid].item()
                    ema_score = self.scores.get(uid, 0)
                    bt.logging.info(
                        f"  UID {uid}: weight={weight:.4f} "
                        f"(score={ema_score:.2f}, stake={stake:.2f}τ, "
                        f"ip={axon.ip}:{axon.port})"
                    )
                except Exception as e:
                    bt.logging.debug(f"  UID {uid}: weight={weight:.4f} (error getting details: {e})")

            if len(sorted_weights) > 10:
                bt.logging.info(f"  ... and {len(sorted_weights) - 10} more miners")

            # Convert to lists
            uids = list(weights_dict.keys())
            weights = [weights_dict[uid] for uid in uids]

            # Convert to tensors
            uids_tensor = torch.tensor(uids, dtype=torch.int64)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)

            bt.logging.info(f"Submitting weights to chain...")
            bt.logging.debug(f"  UIDs: {uids[:20]}{'...' if len(uids) > 20 else ''}")
            bt.logging.debug(f"  Weights sum: {sum(weights):.4f}")

            # Set weights
            success, message = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids_tensor,
                weights=weights_tensor,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )

            if success:
                # Update last submission block
                self.last_weights_block = self.subtensor.block
                bt.logging.success(f"✅ Weights set successfully: {len(uids)} miners")
                bt.logging.info(f"   Transaction broadcast at block {self.last_weights_block}")
            else:
                bt.logging.error(f"❌ Failed to set weights: {message}")

        except Exception as e:
            bt.logging.error(f"Error setting weights: {e}")
            bt.logging.error(traceback.format_exc())

    async def run_step(self):
        """Run one validation step."""
        try:
            # 0. Sync metagraph to get latest miner info
            self.metagraph.sync(subtensor=self.subtensor)

            # Get current block for deterministic consensus
            # All validators at the same block will make the same selections
            current_block = self.subtensor.block
            bt.logging.info(f"🔗 Current block: {current_block}")

            # 1. Select task type (deterministic based on block)
            task_type = self.select_task_type(current_block)
            bt.logging.info(f"🎯 Task type: {task_type}")

            # 2. Create task (deterministic based on block)
            synapse, context = self.create_task(task_type, current_block)

            # 3. Select miners
            # _is_miner_available() excludes invalid axons (0.0.0.0, missing hotkey, etc.)
            all_miners = self.metagraph.axons
            available_miners = [
                (uid, axon) for uid, axon in enumerate(all_miners)
                if uid not in self.blacklist and self._is_miner_available(uid, axon)
            ]

            # Log filtered miners count
            filtered_count = len(all_miners) - len(available_miners) - len(self.blacklist)
            if filtered_count > 0:
                bt.logging.debug(f"Filtered out {filtered_count} invalid axons (0.0.0.0 or invalid config)")

            if not available_miners:
                bt.logging.warning("No available miners")
                await asyncio.sleep(self.query_interval)
                return

            # Select miners using deterministic selection based on block
            # This ensures all validators at the same block select the same miners
            # Query all available miners (no limit)
            # This ensures all miners get scored for fair weight distribution
            num_miners = len(available_miners)

            # Sort miners by: 1) local EMA scores (primary), 2) on-chain incentive (secondary)
            # This allows new miners with 0 incentive to be discovered and scored
            # Note: Initially all scores are 0, so we use deterministic random for fair bootstrapping
            sorted_miners = sorted(
                available_miners,
                key=lambda x: (
                    self.scores.get(x[0], 0),  # Primary: local EMA score
                    self.metagraph.I[x[0]].item() if x[0] < len(self.metagraph.I) else 0,  # Secondary: on-chain
                ),
                reverse=True
            )

            # Log miner selection for debugging
            bt.logging.info(f"🔍 Querying ALL {len(available_miners)} available miners")

            # Query all miners - no top_k/explore_k split needed
            selected_uids = [uid for uid, _ in sorted_miners]
            selected_axons = [axon for _, axon in sorted_miners]

            bt.logging.info(f"📡 Querying {len(selected_axons)} miners: {selected_uids}")

            # Log miner IPs for debugging
            for uid, axon in zip(selected_uids, selected_axons):
                bt.logging.debug(f"  → UID {uid}: {axon.ip}:{axon.port}")

            # 4. Query miners
            with Timer() as t:
                responses = await self.query_miners(synapse, selected_axons)

            bt.logging.info(f"⏱️  Query completed in {t.elapsed:.2f}s")

            # 5. Score responses
            scores = {}
            for uid, response, axon in zip(selected_uids, responses, selected_axons):
                bt.logging.debug(f"\n{'='*60}")
                bt.logging.debug(f"Evaluating Miner UID {uid} ({axon.ip}:{axon.port})")
                bt.logging.debug(f"{'='*60}")

                # Check and log response status (INFO level for operators to see without --logging.debug)
                if response is None:
                    bt.logging.warning(
                        f"⚠️  Miner {uid} ({axon.ip}:{axon.port}): Response is None "
                        f"(connection failed, timeout, or invalid response)"
                    )
                    scores[uid] = 0.0
                    continue

                # Handle both Synapse object and dict response types
                # (Bittensor may return dict instead of deserialized Synapse in some cases)
                if isinstance(response, dict):
                    bt.logging.info(f"📦 Response is dict. Keys: {list(response.keys())}")

                    # Case 1: Response dict has standard synapse fields (output_data, generation_time, etc.)
                    if 'output_data' in response:
                        output_data_val = response.get('output_data')
                        bt.logging.info(f"   Using output_data field from response")
                    # Case 2: Response is the deserialized output_data directly (e.g. {'generated_text': '...'} or task results)
                    else:
                        # The dendrite may return the output_data directly after calling deserialize()
                        output_data_val = response
                        bt.logging.info(f"   Treating entire response dict as output_data")

                    # Convert dict to Synapse for uniform handling
                    response = StoryGenerationSynapse(
                        task_type=response.get('task_type', synapse.task_type),
                        user_input=response.get('user_input', synapse.user_input),
                        blueprint=response.get('blueprint', synapse.blueprint),
                        characters=response.get('characters', synapse.characters),
                        story_arc=response.get('story_arc', synapse.story_arc),
                        chapter_ids=response.get('chapter_ids', synapse.chapter_ids),
                        output_data=output_data_val,
                        generation_time=response.get('generation_time', 0.0),
                        miner_version=response.get('miner_version', 'unknown'),
                        model_info=response.get('model_info', {})
                    )

                if not hasattr(response, 'output_data'):
                    bt.logging.warning(
                        f"⚠️  Miner {uid} ({axon.ip}:{axon.port}): Response missing output_data attribute "
                        f"(response type: {type(response).__name__})"
                    )
                    bt.logging.info(f"   Available attributes: {[a for a in dir(response) if not a.startswith('_')][:10]}")
                    scores[uid] = 0.0
                    continue

                # Extract response data for logging
                output_data = response.output_data
                generation_time = getattr(response, 'generation_time', 0.0)
                miner_version = getattr(response, 'miner_version', 'unknown')
                model_info = getattr(response, 'model_info', {})

                if output_data is None:
                    bt.logging.warning(
                        f"⚠️  Miner {uid} ({axon.ip}:{axon.port}): output_data is None "
                        f"(gen_time: {generation_time}s)"
                    )
                    scores[uid] = 0.0
                    continue

                # Additional debug details (only with --logging.debug)
                bt.logging.debug(f"Response type: {type(response)}")
                bt.logging.debug(f"output_data type: {type(output_data)}")
                bt.logging.debug(f"output_data value: {output_data}")

                # Log response metadata
                bt.logging.debug(f"Response metadata:")
                bt.logging.debug(f"  - Generation time: {generation_time:.2f}s")
                bt.logging.debug(f"  - Miner version: {miner_version}")
                bt.logging.debug(f"  - Model info: {model_info}")

                # Log output data preview
                output_preview = str(output_data)[:200] + "..." if len(str(output_data)) > 200 else str(output_data)
                bt.logging.debug(f"  - Output preview: {output_preview}")

                # Check plagiarism
                is_plagiarism, plag_type, similarity = self.detect_plagiarism(response, responses)

                if is_plagiarism:
                    bt.logging.warning(f"⚠️  Miner {uid}: Plagiarism detected ({plag_type}, similarity={similarity:.2f})")
                    self.violations[uid] = self.violations.get(uid, 0) + 1

                    if self.violations[uid] >= 3:
                        self.blacklist.add(uid)
                        bt.logging.error(f"🚫 Miner {uid} blacklisted (3 violations)")

                    scores[uid] = 0.0
                    continue

                # Score response
                score, breakdown = self.score_response(response, context)
                scores[uid] = score

                # Detailed scoring breakdown
                bt.logging.info(f"📊 Miner {uid} ({axon.ip}): {score:.2f} points")
                bt.logging.debug(f"  Technical Score: {breakdown['technical']:.1f}/30")
                if 'technical_breakdown' in breakdown:
                    for k, v in breakdown['technical_breakdown'].items():
                        bt.logging.debug(f"    - {k}: {v}")

                bt.logging.debug(f"  Structure Score: {breakdown['structure']:.1f}/40")
                if 'structure_breakdown' in breakdown:
                    for k, v in breakdown['structure_breakdown'].items():
                        bt.logging.debug(f"    - {k}: {v}")

                bt.logging.debug(f"  Content Score: {breakdown['content']:.1f}/30")
                if 'content_breakdown' in breakdown:
                    for k, v in breakdown['content_breakdown'].items():
                        bt.logging.debug(f"    - {k}: {v}")

                bt.logging.debug(f"  Base Score: {breakdown.get('base_score', 0):.2f}/100")

                if 'model_quality' in breakdown:
                    bt.logging.debug(f"  Model Quality Multiplier: {breakdown['model_quality'].get('final_multiplier', 1.0):.2f}x")

                bt.logging.debug(f"  Final Score: {score:.2f}")

                # Record history (v3.0.0: store output_data instead of output_json)
                self.history.append({
                    "uid": uid,
                    "task_type": task_type,
                    "output_data": response.output_data,
                    "score": score,
                    "timestamp": time.time()
                })

            # 6. Update EMA scores
            self.update_ema_scores(scores)

            # 7. Update statistics
            self.total_queries += 1
            self.successful_queries += len([s for s in scores.values() if s > 0])
            self.total_rewards += sum(scores.values())

            # 8. Set weights (rate limited by blocks, not query count)
            # Try to set weights every step, let can_set_weights() handle rate limiting
            if self.can_set_weights():
                bt.logging.info(f"\n{'='*60}")
                bt.logging.info(f"Setting weights (query #{self.total_queries})")
                bt.logging.info(f"{'='*60}")
                await self.set_weights()

        except Exception as e:
            bt.logging.error(f"Error in run_step: {e}")
            bt.logging.error(traceback.format_exc())

    async def run(self):
        """Main run loop."""
        bt.logging.info("🚀 Starting validator...")

        # Sync metagraph
        self.metagraph.sync(subtensor=self.subtensor)
        bt.logging.info(f"📡 Metagraph synced: {len(self.metagraph.axons)} miners")

        try:
            while True:
                await self.run_step()

                # Print statistics
                if self.total_queries % 10 == 0:
                    avg_score = (
                        self.total_rewards / self.successful_queries
                        if self.successful_queries > 0
                        else 0.0
                    )

                    bt.logging.info(
                        f"\n📈 Statistics:\n"
                        f"  Total queries: {self.total_queries}\n"
                        f"  Successful: {self.successful_queries}\n"
                        f"  Avg score: {avg_score:.2f}\n"
                        f"  Blacklisted: {len(self.blacklist)}\n"
                        f"  Active miners: {len(self.scores)}"
                    )

                # Wait before next query
                await asyncio.sleep(self.query_interval)

        except KeyboardInterrupt:
            bt.logging.info("🛑 Shutting down validator...")

            # Final weight update
            await self.set_weights()


def get_config():
    """Get configuration from command line arguments."""
    parser = argparse.ArgumentParser()

    # Add Bittensor standard arguments
    bt.Subtensor.add_args(parser)
    bt.Wallet.add_args(parser)
    bt.logging.add_args(parser)

    # Add custom arguments
    parser.add_argument("--netuid", type=int, default=92, help="Subnet netuid (StoryNet subnet ID)")

    # Parse and add bittensor config
    config = bt.Config(parser)

    return config


def main():
    """Main entry point."""
    config = get_config()

    # Setup logging
    bt.logging.set_trace(config.logging.debug)
    bt.logging.set_debug(config.logging.debug)
    bt.logging.set_info(config.logging.info)

    # Create and run validator
    validator = StoryValidator(config)
    asyncio.run(validator.run())


if __name__ == "__main__":
    main()
