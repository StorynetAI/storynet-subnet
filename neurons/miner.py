"""
StoryNet Bittensor Miner
=======================

Listens for story generation requests from Validators and responds
with AI-generated content.

Usage:
    python neurons/miner.py \
        --netuid 92 \
        --wallet.name my_miner \
        --wallet.hotkey default \
        --logging.info
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from typing import Dict, Any, Optional, Tuple

import bittensor as bt
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.protocol import StoryGenerationSynapse
from template.utils import Timer, compute_hash
from generators.loader import GeneratorLoader

# Load environment variables
load_dotenv()


class StoryMiner:
    """
    StoryNet Miner that generates stories using configurable backends.

    The miner:
    1. Listens for requests from Validators via Bittensor network
    2. Processes 4 types of tasks: blueprint, characters, story_arc, chapters
    3. Uses GeneratorLoader to support multiple generation methods
    4. Returns JSON-formatted story content

    """

    def __init__(self, config: bt.Config):
        """
        Initialize the miner.

        Args:
            config: Bittensor configuration object
        """
        self.config = config
        bt.logging.info("Initializing StoryNet Miner...")

        # Initialize Bittensor components
        self.wallet = bt.Wallet(config=self.config)
        self.subtensor = bt.Subtensor(config=self.config)
        self.metagraph = bt.Metagraph(netuid=self.config.netuid, network=self.subtensor.network)

        # Initialize Generator (replaces hardcoded OpenAI)
        bt.logging.info("Loading story generator...")
        self.generator = GeneratorLoader()

        generator_mode = self.generator.get_mode()
        model_info = self.generator.get_model_info()

        bt.logging.info(f"Generator Mode: {generator_mode}")
        bt.logging.info(f"Model: {model_info.get('name', 'unknown')}")

        # Statistics
        self.requests_processed = 0
        self.total_generation_time = 0.0
        self.errors = 0

        bt.logging.info(f"✅ Wallet: {self.wallet.hotkey.ss58_address}")
        bt.logging.info(f"✅ Netuid: {self.config.netuid}")

    def setup_axon(self):
        """Setup and start the axon server."""
        bt.logging.info("Setting up axon...")

        self.axon = bt.Axon(wallet=self.wallet, config=self.config)

        # Override external IP/port if specified in config (for NAT/tunnel scenarios)
        try:
            ext_ip = getattr(self.config.axon, 'external_ip', None)
            ext_port = getattr(self.config.axon, 'external_port', None)
            if ext_ip:
                bt.logging.info(f"🔧 Setting external IP to: {ext_ip}")
                self.axon.external_ip = ext_ip
            if ext_port:
                bt.logging.info(f"🔧 Setting external port to: {ext_port}")
                self.axon.external_port = int(ext_port)
        except Exception as e:
            bt.logging.warning(f"Could not set external IP/port: {e}")

        # Log axon info for debugging
        bt.logging.info(f"📡 Axon IP: {self.axon.external_ip}")
        bt.logging.info(f"📡 Axon Port: {self.axon.external_port}")

        # Attach forward function
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority
        )

        # Start axon
        self.axon.start()

        # Register to network
        self.subtensor.serve_axon(
            netuid=self.config.netuid,
            axon=self.axon
        )

        bt.logging.info(f"✅ Axon registered: {self.axon.external_ip}:{self.axon.external_port}")
        bt.logging.info(f"✅ Registered to subnet {self.config.netuid}")

    async def forward(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
        """
        Process incoming request from Validator.

        Args:
            synapse: Request synapse containing task information

        Returns:
            Synapse with generated content filled in
        """
        try:
            bt.logging.info(f"📨 Received {synapse.task_type} request from {synapse.validator_hotkey}")
            bt.logging.debug(f"Input data: user_input='{synapse.user_input[:100]}...', blueprint_keys={list(synapse.blueprint.keys()) if synapse.blueprint else 'None'}, "
                             f"characters_len={len(synapse.characters) if synapse.characters else 0}, "
                             f"story_arc_keys={list(synapse.story_arc.keys()) if synapse.story_arc else 'None'}, "
                             f"chapter_ids={synapse.chapter_ids}")

            with Timer() as t:
                # Build input_data from synapse fields (Protocol v3.1.0)
                input_data = {
                    "user_input": synapse.user_input,
                    "blueprint": synapse.blueprint,
                    "characters": synapse.characters,
                    "story_arc": synapse.story_arc,
                    "chapter_ids": synapse.chapter_ids,
                    "task_type": synapse.task_type  # Pass task type to generator
                }

                # Use unified generator (supports local models, APIs, etc.)
                bt.logging.debug(f"Calling generator with task_type: {synapse.task_type}")
                result = await self.generator.generate(input_data)
                bt.logging.debug(f"Generator result type: {type(result)}, keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}")

                # Extract generated content from generator response
                generated_content = result.get("generated_content", "")
                bt.logging.debug(f"Generated content length: {len(generated_content)}, preview: '{generated_content[:200]}...'")

                # Try to parse as JSON if task expects structured output
                try:
                    if generated_content:
                        # Clean markdown wrappers if present
                        content = generated_content.strip()
                        
                        # Handle different types of code blocks
                        if content.startswith("```json"):
                            # Extract content from ```json code blocks
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif content.startswith("```"):
                            # Extract content from generic code blocks
                            content = content.split("```")[1].split("```")[0].strip()
                        
                        # Parse JSON
                        output_data = json.loads(content)
                        bt.logging.debug(f"Parsed output_data type: {type(output_data)}, keys: {output_data.keys() if isinstance(output_data, dict) else 'N/A'}")

                        # Log the actual generated content for comparison with what validator receives
                        bt.logging.success(f"🎯 Generated {synapse.task_type} result: {len(json.dumps(output_data))} chars")
                        bt.logging.trace(f"Generated content: {json.dumps(output_data)[:500]}...")  # Truncate for log

                        # Validate format matches task type (Protocol v3.2.0)
                        # ALL tasks must return Dict (JSON object), never List (JSON array)
                        # - blueprint: returns {title, genre, setting, ...}
                        # - characters: returns {characters: [...]}
                        # - story_arc: returns {title, chapters, arcs, ...}
                        # - chapters: returns {chapters: [...]}
                        if isinstance(output_data, list):
                            bt.logging.warning(
                                f"⚠️  {synapse.task_type} task returned array instead of object. "
                                f"LLM misunderstood the prompt format."
                            )
                            # Wrap in error object with helpful context
                            output_data = {
                                "error": f"Format mismatch: {synapse.task_type} must return JSON object, not array",
                                "hint": "Check your prompt templates - all tasks require object format {...}",
                                "raw_output": output_data
                            }
                    else:
                        bt.logging.warning(f"No generated content from generator, returning error object")
                        output_data = {"error": "Empty response from generator"}
                except json.JSONDecodeError as e:
                    bt.logging.info(f"Generated content is not JSON (this is expected for story content): {e}")
                    # For story generation tasks, non-JSON content is normal
                    # Try to extract structured data from markdown content
                    if generated_content:
                        # For story tasks, return the content in a structured format
                        output_data = {
                            "generated_text": generated_content,
                            "format": "text",
                            "task_type": synapse.task_type,
                            "content_preview": generated_content[:200] + "..." if len(generated_content) > 200 else generated_content
                        }
                        bt.logging.debug(f"Returning story content as structured output_data: {type(output_data)}")
                    else:
                        output_data = {"error": "Empty response from generator"}

            # Fill response fields (Protocol v3.2.0)
            synapse.output_data = output_data
            synapse.generation_time = t.elapsed
            synapse.miner_version = "2.0.0"  # Updated version with flexible generators

            # Populate model_info for transparency (Protocol v3.2.0)
            model_info = {
                "mode": self.generator.get_mode(),
                "name": self.generator.get_model_info().get("name", "unknown"),
                "version": self.generator.get_model_info().get("version"),
                "provider": self.generator.get_model_info().get("provider"),
                "parameters": self.generator.get_model_info().get("parameters", {})
            }
            synapse.model_info = model_info

            # WORKAROUND: Also embed model_info in output_data since Bittensor doesn't transmit model_info field
            if isinstance(output_data, dict):
                output_data["_model_info"] = model_info
                synapse.output_data = output_data

            # Update statistics
            self.requests_processed += 1
            self.total_generation_time += t.elapsed

            output_size = len(json.dumps(output_data, ensure_ascii=False)) if output_data else 0
            bt.logging.success(
                f"✅ Generated {synapse.task_type} in {t.elapsed:.2f}s "
                f"(output: {output_size} chars)"
            )
            bt.logging.info(f"📋 Model info set: {synapse.model_info}")
            bt.logging.debug(f"Sending output_data: {type(output_data)} with keys: {output_data.keys() if isinstance(output_data, dict) else 'N/A'}")

            # Debug: Check if model_info is included in serialization
            try:
                synapse_dump = synapse.model_dump()
                bt.logging.info(f"🔍 Synapse model_dump() model_info: {synapse_dump.get('model_info')}")
                bt.logging.debug(f"🔍 Synapse model_dump() keys: {list(synapse_dump.keys())}")
            except Exception as e:
                bt.logging.warning(f"⚠️ Error dumping synapse: {e}")

            # Double-check that output_data was set correctly
            if not synapse.output_data:
                bt.logging.error("❌ ERROR: synapse.output_data is empty after assignment!")
            else:
                bt.logging.success(f"🎯 Synapse output_data set with {len(json.dumps(synapse.output_data))} chars")

            return synapse

        except Exception as e:
            self.errors += 1
            bt.logging.error(f"❌ Error processing request: {e}")
            bt.logging.error(traceback.format_exc())

            synapse.output_data = {"error": str(e)}
            synapse.generation_time = 0.0
            synapse.miner_version = "2.0.0"

            # Include model_info even in error case for transparency
            synapse.model_info = {
                "mode": self.generator.get_mode(),
                "name": self.generator.get_model_info().get("name", "unknown"),
                "version": self.generator.get_model_info().get("version"),
                "provider": self.generator.get_model_info().get("provider"),
                "parameters": self.generator.get_model_info().get("parameters", {})
            }

            bt.logging.warning(f"Sending error response: {synapse.output_data}")

            return synapse

    # Note: All generation logic is now handled by GeneratorLoader
    # No need for task-specific generate_* functions
    # The generator handles prompt building based on task_type

    def blacklist(self, synapse: StoryGenerationSynapse) -> Tuple[bool, str]:
        """
        Determine if request should be blacklisted.

        Args:
            synapse: Incoming request

        Returns:
            Tuple of (should_blacklist, reason)
        """
        # Accept all requests for now
        # Can add blacklisting logic later (e.g., known malicious validators)
        return False, ""

    def priority(self, synapse: StoryGenerationSynapse) -> float:
        """
        Determine priority for request processing.

        Args:
            synapse: Incoming request

        Returns:
            Priority score (higher = more priority)
        """
        # Give higher priority to validators with higher stake
        validator_hotkey = synapse.validator_hotkey
        if validator_hotkey and validator_hotkey in self.metagraph.hotkeys:
            uid = self.metagraph.hotkeys.index(validator_hotkey)
            stake = self.metagraph.S[uid].item()
            return stake
        return 0.0

    async def run(self):
        """Main run loop."""
        bt.logging.info("🚀 Starting miner...")

        # Setup axon
        self.setup_axon()

        # Keep alive and print stats
        try:
            while True:
                await asyncio.sleep(60)

                # Print statistics
                avg_time = (
                    self.total_generation_time / self.requests_processed
                    if self.requests_processed > 0
                    else 0.0
                )

                bt.logging.info(
                    f"📊 Stats: "
                    f"Requests={self.requests_processed}, "
                    f"AvgTime={avg_time:.2f}s, "
                    f"Errors={self.errors}"
                )

                # Resync metagraph
                self.metagraph.sync(subtensor=self.subtensor)

        except KeyboardInterrupt:
            bt.logging.info("🛑 Shutting down miner...")
            self.axon.stop()


def get_config():
    """Get configuration from command line arguments."""
    parser = argparse.ArgumentParser()

    # Bittensor arguments
    parser.add_argument("--netuid", type=int, default=92, help="Subnet netuid (StoryNet subnet ID)")
    parser.add_argument("--wallet.name", type=str, default="miner", help="Wallet name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", help="Wallet hotkey")
    parser.add_argument("--subtensor.network", type=str, default="finney", help="Bittensor network (finney=mainnet, test=testnet)")
    parser.add_argument("--subtensor.chain_endpoint", type=str, default=None, help="Subtensor chain endpoint")
    parser.add_argument("--logging.info", action="store_true", help="Enable info logging")
    parser.add_argument("--logging.debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--axon.port", type=int, default=8091, help="Axon port")
    parser.add_argument("--axon.external_ip", type=str, default=None, help="External IP address (required for cloud/NAT servers)")
    parser.add_argument("--axon.external_port", type=int, default=None, help="External port (if different from axon.port)")

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

    # Create and run miner
    miner = StoryMiner(config)
    asyncio.run(miner.run())


if __name__ == "__main__":
    main()
