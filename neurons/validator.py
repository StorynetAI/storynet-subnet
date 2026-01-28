"""
StoryNet Bittensor Validator
============================

This validator:
1. Generates tasks for miners
2. Queries miners with different task types
3. Gets scores from external API
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

# Import API client
import requests
import json
from typing import Dict, Any, List

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

    def __init__(self, config=None):
        """Initialize the validator."""
        self.config = config or self.get_config()
        
        # 如果api_endpoint未设置，则使用默认值
        if not hasattr(self.config, 'api_endpoint') or self.config.api_endpoint is None:
            self.config.api_endpoint = "https://api.storyai.art"
        
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
        # 修改权重更新频率，从固定100改为可配置参数，默认为10
        self.weight_update_frequency = int(os.getenv("WEIGHT_UPDATE_FREQUENCY", "10"))

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
        self.history = {}  # {miner_uid: {'scores': deque, 'timestamps': deque}}
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

        # State file path for persistence
        self.state_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "validator_state.json"
        )

        # Load persisted state (EMA scores, violations, etc.)
        self._load_state()

        bt.logging.info(f"✅ Wallet: {self.wallet.hotkey.ss58_address}")
        bt.logging.info(f"✅ Netuid: {self.config.netuid}")
        bt.logging.info(f"✅ Query interval: {self.query_interval}s")


    def _load_state(self):
        """
        Load persisted validator state from file.

        This ensures EMA scores and other state survive validator restarts.
        Without this, all miner scores would reset to 0 on restart.
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # Restore EMA scores (convert string keys back to int)
                self.scores = {int(k): v for k, v in state.get("scores", {}).items()}

                # Restore history data
                history_data = state.get("history", {})
                self.history = {}
                for uid_str, hist_data in history_data.items():
                    uid = int(uid_str)
                    self.history[uid] = {
                        'scores': deque(hist_data['scores'], maxlen=100),  # Keep last 100 scores
                        'timestamps': deque(hist_data['timestamps'], maxlen=100)  # Keep timestamps
                    }

                # Restore violations count
                self.violations = {int(k): v for k, v in state.get("violations", {}).items()}

                # Restore blacklist
                self.blacklist = set(state.get("blacklist", []))

                # Restore last weights block
                self.last_weights_block = state.get("last_weights_block", 0)

                # Restore statistics
                self.total_queries = state.get("total_queries", 0)
                self.successful_queries = state.get("successful_queries", 0)
                self.total_rewards = state.get("total_rewards", 0.0)

                bt.logging.success(
                    f"✅ Loaded validator state from {self.state_file}: "
                    f"{len(self.scores)} miner scores, "
                    f"{len(self.blacklist)} blacklisted, "
                    f"last_weights_block={self.last_weights_block}"
                )
            else:
                bt.logging.info(f"📝 No state file found at {self.state_file}, starting fresh")
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to load state file: {e}, starting fresh")

    def _save_state(self):
        """
        Save validator state to file for persistence across restarts.

        This is called after updating EMA scores and setting weights
        to ensure state is not lost on restart.
        """
        try:
            state = {
                "scores": self.scores,
                "history": {
                    uid: {
                        "scores": list(hist['scores']),
                        "timestamps": list(hist['timestamps'])
                    }
                    for uid, hist in self.history.items()
                },
                "violations": self.violations,
                "blacklist": list(self.blacklist),
                "last_weights_block": self.last_weights_block,
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "total_rewards": self.total_rewards,
                "saved_at": time.time()
            }

            # Write to temp file first, then rename (atomic operation)
            temp_file = self.state_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)

            # Atomic rename
            os.replace(temp_file, self.state_file)

            bt.logging.debug(f"💾 Saved validator state: {len(self.scores)} miner scores")
        except Exception as e:
            bt.logging.error(f"❌ Failed to save state: {e}")

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
            
            # Debug: log the actual responses received
            for i, response in enumerate(responses):
                bt.logging.debug(f"Raw response from miner {i}: type={type(response)}")
                if hasattr(response, '__dict__'):
                    bt.logging.debug(f"Response {i} attributes: {list(response.__dict__.keys())}")
                if isinstance(response, dict):
                    bt.logging.debug(f"Response {i} dict keys: {list(response.keys())}")
                    if 'output_data' in response:
                        output_size = len(json.dumps(response['output_data'], ensure_ascii=False)) if response['output_data'] else 0
                        bt.logging.debug(f"Response {i} output_data size: {output_size}")
                elif hasattr(response, 'output_data'):
                    output_size = len(json.dumps(response.output_data, ensure_ascii=False)) if response.output_data else 0
                    bt.logging.debug(f"Response {i} has output_data attribute, size: {output_size}")
                else:
                    bt.logging.debug(f"Response {i} content: {str(response)[:200] if response else 'None'}")
            
            # Process responses to ensure they are StoryGenerationSynapse objects
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, StoryGenerationSynapse):
                    # Direct StoryGenerationSynapse object - use as-is
                    bt.logging.debug(f"Response {i} is StoryGenerationSynapse object")
                    processed_responses.append(response)
                elif isinstance(response, dict):
                    # Handle dict response format - extract fields from the dictionary
                    # Create a new StoryGenerationSynapse instance and populate it with the data
                    synapse_obj = StoryGenerationSynapse()
                    
                    # Log all available keys in the response dict
                    bt.logging.debug(f"Processing dict response {i} with keys: {list(response.keys())}")
                    
                    # Extract all fields from the dict response to the synapse object
                    for key, value in response.items():
                        if hasattr(synapse_obj, key):
                            setattr(synapse_obj, key, value)
                            bt.logging.debug(f"Set {key} = {value}")
                    
                    # Ensure all required fields are populated
                    if not hasattr(synapse_obj, 'output_data') or getattr(synapse_obj, 'output_data', None) is None:
                        synapse_obj.output_data = response
                        bt.logging.debug(f"Used entire response as output_data for {i}")
                    
                    # Debug: log what we extracted
                    output_size = len(json.dumps(synapse_obj.output_data, ensure_ascii=False)) if synapse_obj.output_data else 0
                    bt.logging.debug(f"Processed dict response {i}: output_data_size={output_size}, gen_time={synapse_obj.generation_time}, model_info={synapse_obj.model_info}")
                    processed_responses.append(synapse_obj)
                else:
                    # Fallback: create empty synapse if response type is unexpected
                    bt.logging.warning(f"Unexpected response type from miner {i}: {type(response)}")
                    synapse_obj = StoryGenerationSynapse()
                    synapse_obj.output_data = {"error": "Invalid response type received"}
                    processed_responses.append(synapse_obj)
            
            return processed_responses
        except Exception as e:
            bt.logging.error(f"Error querying miners: {e}")
            # Return empty synapse objects in case of error
            return [StoryGenerationSynapse()] * len(miners)

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
        """Update EMA scores for miners and record history."""
        for uid, score in new_scores.items():
            if uid not in self.scores:
                self.scores[uid] = score
            else:
                self.scores[uid] = exponential_moving_average(
                    score,
                    self.scores[uid],
                    self.ema_alpha
                )
            
            # Update history data
            if uid not in self.history:
                self.history[uid] = {
                    'scores': deque(maxlen=100),
                    'timestamps': deque(maxlen=100)
                }
            
            # Add current score and timestamp to history
            self.history[uid]['scores'].append(score)
            self.history[uid]['timestamps'].append(time.time())

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
        Calculate weights using updated system:
        - 0% Stake weight (removed to eliminate stake influence)
        - 60% Quality score (current表现)
        - 25% Historical score (长期稳定性)
        - 15% removed from original stake component
        The composite score is calculated as 60% quality + 25% historical of individual miner's scores
        """
        if not self.scores:
            return {}

        # Get burn_uid (subnet owner) to exclude from miner weights
        burn_uid = self.get_burn_uid()

        # Calculate composite scores without stake consideration
        # IMPORTANT: Exclude burn_uid (owner) from miner scoring
        composite_scores = {}

        for uid, quality_score in self.scores.items():
            # Skip burn_uid - owner is not a miner
            if uid == burn_uid:
                bt.logging.debug(f"Skipping burn_uid {uid} from miner scoring")
                continue

            # 1. Normalize quality score to 0-1
            normalized_quality = quality_score / 100.0

            # 2. Calculate historical score component
            # Using miner's historical performance stored in self.history
            # If no history exists, default to average quality score
            historical_score = 0.5  # Default to 50% if no historical data
            
            if uid in self.history and len(self.history[uid]['scores']) > 0:
                # Calculate average of historical scores
                hist_avg = sum(self.history[uid]['scores']) / len(self.history[uid]['scores'])
                historical_score = hist_avg / 100.0  # Normalize to 0-1

            # 3. Composite score based on new distribution:
            # Original: 15% stake + 75% quality + 10% historical (total 100% of scoring components)
            # New: 0% stake + 75% quality + 25% historical (total 100% of scoring components)
            composite = (
                0.75 * normalized_quality +      # 75% from quality (API obtained scores)
                0.25 * historical_score          # 25% from historical
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
            for uid, weight in sorted_weights:  # 显示所有权重，不只是前10个
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

            # Remove the "... and X more miners" message since we're showing all now
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

            # Initialize step counter if not exists
            if not hasattr(self, 'step'):
                self.step = 0

            # 1. Deterministically select task type based on block number
            task_type = self.select_task_type(current_block)
            
            # 2. Create task with deterministic context
            synapse, context = self.create_task(task_type, current_block)
            context["task_type"] = task_type
            
            # 3. Select available miners to query
            available_axons = []
            selected_uids = []
            
            for uid in range(len(self.metagraph.axons)):
                axon = self.metagraph.axons[uid]
                if self._is_miner_available(uid, axon):
                    available_axons.append(axon)
                    selected_uids.append(uid)
            
            # Limit number of concurrent queries
            if len(selected_uids) > self.config.num_concurrent_forwards:
                # Select miners randomly but deterministically based on block
                random.seed(current_block)
                indices = list(range(len(selected_uids)))
                random.shuffle(indices)
                selected_indices = sorted(indices[:self.config.num_concurrent_forwards])
                
                selected_uids = [selected_uids[i] for i in selected_indices]
                available_axons = [available_axons[i] for i in selected_indices]

            if not selected_uids:
                bt.logging.warning("No available miners to query")
                return

            bt.logging.info(f"📨 Querying {len(selected_uids)} miners for {task_type} task")
            
            # 4. Query miners with the task
            try:
                bt.logging.debug(f"📡 Querying miners: {len(available_axons)} available")
                for i, axon in enumerate(available_axons):
                    bt.logging.debug(f"   [{i}] UID {selected_uids[i]}: {axon.ip}:{axon.port} ({axon.hotkey[:8]}...)")
                
                responses = await self.query_miners(synapse, available_axons)
                
                # Log detailed response information
                bt.logging.debug(f"📥 Received {len(responses)} responses from miners")
                for i, response in enumerate(responses):
                    uid = selected_uids[i]

                    # Debug: Log raw response type and attributes for model_info investigation
                    bt.logging.debug(f"   UID {uid} raw response type: {type(response)}")
                    if hasattr(response, 'model_info'):
                        bt.logging.debug(f"   UID {uid} response.model_info (attr): {response.model_info}")
                    if hasattr(response, 'model_dump'):
                        try:
                            dumped = response.model_dump()
                            bt.logging.debug(f"   UID {uid} model_dump keys: {list(dumped.keys())}")
                            bt.logging.debug(f"   UID {uid} model_dump model_info: {dumped.get('model_info')}")
                        except Exception as e:
                            bt.logging.debug(f"   UID {uid} model_dump error: {e}")

                    # Check if response is a dictionary or object
                    if isinstance(response, dict):
                        output_data = response.get('output_data')
                        generation_time = response.get('generation_time', 0.0)
                        model_info = response.get('model_info', {})
                    else:
                        output_data = getattr(response, 'output_data', None)
                        generation_time = getattr(response, 'generation_time', 0.0)
                        model_info = getattr(response, 'model_info', {})

                    output_size = len(str(output_data)) if output_data else 0
                    bt.logging.debug(f"   UID {uid}: output_data size={output_size}, gen_time={generation_time:.2f}s, model_info={model_info}")
                    
                    if not output_data or output_data == {}:
                        bt.logging.warning(f"   UID {uid}: ⚠️ Empty output_data received!")
                    else:
                        bt.logging.info(f"   UID {uid}: ✅ Valid output_data received (size: {output_size})")
                        
            except Exception as e:
                bt.logging.error(f"❌ Error querying miners: {e}")
                responses = [synapse] * len(available_axons)

            # 5. Get scores from external API
            scores = {}
            try:
                scores = self.get_scores_from_api(
                    responses=responses,
                    context=context,
                    miner_uids=selected_uids
                )
                bt.logging.info(f"✅ Got scores from API for {len(scores)} miners")
            except Exception as e:
                bt.logging.error(f"❌ Error getting scores from API: {e}")
                # Fallback to zero scores if API fails
                scores = {uid: 0.0 for uid in selected_uids}

            # 6. Update EMA scores for all queried miners
            self.update_ema_scores(scores)

            # 7. Periodically set weights on chain
            if self.step > 0 and self.step % self.weight_update_frequency == 0:
                await self.set_weights()

            # 10. Log step results
            avg_score = sum(scores.values()) / len(scores) if scores else 0.0
            bt.logging.info(
                f"📊 Step {self.step} completed | Task: {task_type} | "
                f"Miners: {len(selected_uids)} | Avg score: {avg_score:.2f}"
            )

            # 11. Update statistics
            self.total_queries += 1
            self.successful_queries += len([s for s in scores.values() if s > 0])
            self.total_rewards += sum(scores.values())

            # 12. Increment step counter
            self.step += 1
            self.last_updated_block = current_block
        except Exception as e:
            bt.logging.error(f"Error in run_step: {e}")
            bt.logging.error(traceback.format_exc())
            # Re-raise the exception to prevent silent failures
            raise

    def get_scores_from_api(
        self,
        responses: List[StoryGenerationSynapse],
        context: Dict[str, Any],
        miner_uids: List[int]
    ) -> Dict[int, float]:
        """
        从API获取矿工评分 - 批量请求版本
        """
        # 准备要发送到API的数据
        synapse_data_list = []
        
        for i, response in enumerate(responses):
            uid = miner_uids[i] if i < len(miner_uids) else -1
            
            # 修复处理response可能为字典的问题
            # 注意：需要正确处理 None 值（属性存在但值为 None 的情况）
            if isinstance(response, dict):
                output_data = response.get('output_data')
                generation_time = response.get('generation_time')
                task_type = response.get('task_type')
                model_info = response.get('model_info')
            else:
                output_data = getattr(response, 'output_data', None)
                generation_time = getattr(response, 'generation_time', None)
                task_type = getattr(response, 'task_type', None)
                model_info = getattr(response, 'model_info', None)
            
            # 处理可能的None值（根据数据完整性规范，保留原始值，仅当值为None时提供默认值）
            if generation_time is None:
                generation_time = 0.0
            if task_type is None:
                task_type = ''
            
            output_size = len(str(output_data)) if output_data else 0
            bt.logging.debug(f"UID {uid} API prep: output_data size={output_size}, type={type(output_data)}")
            bt.logging.debug(f"UID {uid} model_info: {model_info}")
            
            # WORKAROUND: If model_info is None, try to extract from output_data._model_info
            # This is because Bittensor doesn't transmit model_info field properly
            if model_info is None and isinstance(output_data, dict):
                model_info = output_data.get("_model_info")
                if model_info:
                    bt.logging.info(f"UID {uid} model_info extracted from output_data._model_info: {model_info}")
                else:
                    bt.logging.warning(f"UID {uid} model_info is None and not found in output_data")

            synapse_data_list.append({
                "output_data": output_data,
                "generation_time": generation_time,
                "task_type": task_type,
                "model_info": model_info
            })
        
        # 发送批量请求到API - 一次性为所有矿工评分
        scores = {}
        
        # 准备批量请求数据
        api_data = {
            "netuid": self.config.netuid,
            "hotkeys": [self.metagraph.hotkeys[uid] for uid in miner_uids],
            "uids": miner_uids,
            "task_type": context.get("task_type", ""),
            "responses": synapse_data_list
        }
        
        # 记录批量请求日志
        bt.logging.info(f"📤 Sending bulk request to API for {len(miner_uids)} miners: netuid={self.config.netuid}")
        bt.logging.debug(f"Bulk request data: {api_data}")
        
        # 发送批量请求到API
        try:
            api_url = f"{self.config.api_endpoint}/score-miners/"
            bt.logging.info(f"Attempting to connect to API at {api_url} for {len(miner_uids)} miners")
            
            response = requests.post(api_url, json=api_data, timeout=120)  # 增加超时时间以适应批量请求
            
            bt.logging.info(f"📥 Received bulk API response: Status {response.status_code}")
            
            if response.status_code == 200:
                scores_data = response.json()
                bt.logging.info(f"📥 Bulk API returned {len(scores_data)} score records")
                
                # 将API返回的评分转换为字典格式
                for score_item in scores_data:
                    score_uid = score_item.get('uid', 0)
                    score = score_item.get('score', 0.0)
                    breakdown = score_item.get('breakdown', {})
                    timestamp = score_item.get('timestamp', 'N/A')
                    
                    scores[score_uid] = score
                    bt.logging.info(f"   UID {score_uid}: score={score:.2f}, timestamp={timestamp}")
                    bt.logging.debug(f"     Breakdown: tech={breakdown.get('technical', 0)}, struct={breakdown.get('structure', 0)}, content={breakdown.get('content', 0)}, narrative={breakdown.get('narrative', 0)}")
            else:
                bt.logging.error(f"Bulk API request failed with status {response.status_code}: {response.text}")
                # 为所有矿工返回默认评分
                for uid in miner_uids:
                    scores[uid] = 0.0
                    
        except Exception as e:
            bt.logging.error(f"Error getting scores from bulk API request: {e}")
            # 为所有矿工返回默认评分
            for uid in miner_uids:
                scores[uid] = 0.0
        
        bt.logging.info(f"Validator received scores for UIDs: {list(scores.keys())}")
        return scores

    def get_config(self):
        """Get configuration from command line arguments."""
        parser = argparse.ArgumentParser()

        # Add Bittensor standard arguments
        bt.Subtensor.add_args(parser)
        bt.Wallet.add_args(parser)
        bt.logging.add_args(parser)

        # Add custom arguments
        parser.add_argument("--netuid", type=int, default=92, help="Subnet netuid (StoryNet subnet ID)")
        parser.add_argument("--num_concurrent_forwards", type=int, default=255, help="Number of concurrent forwards")
        parser.add_argument("--timeout", type=int, default=30, help="Query timeout in seconds")
        parser.add_argument("--weight_update_interval", type=int, default=100, help="How often to update weights")
        # 添加一个选项来指定API端点
        parser.add_argument("--api_endpoint", type=str, default="https://api.storyai.art", 
                           help="API endpoint for scoring")

        return parser.parse_args()

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

            # Save final state before shutdown
            self._save_state()
            bt.logging.info("💾 Final state saved")

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
    parser.add_argument("--num_concurrent_forwards", type=int, default=255, help="Number of concurrent forwards")
    parser.add_argument("--timeout", type=int, default=30, help="Query timeout in seconds")
    parser.add_argument("--weight_update_interval", type=int, default=100, help="How often to update weights")

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
