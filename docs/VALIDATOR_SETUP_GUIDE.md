# StoryNet (SN92) Validator Setup Guide

> Last Updated: 2025-01-12

---

## Quick Info

| Item | Value |
|------|-------|
| **Subnet ID** | 92 |
| **Network** | Finney (Mainnet) |
| **GitHub** | https://github.com/StorynetAI/storynet-subnet |
| **Protocol** | 3.2.1 |

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| vCPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Storage | 20 GB SSD | 50 GB SSD |
| GPU | Not required | Not required |
| Network | Port 19292 open | Low latency |

---

## LLM for Narrative Scoring (Recommended)

The validator uses AI to evaluate story quality (30 pts out of 100). It auto-detects any OpenAI-compatible API.

> ⚠️ **Important for Consensus**: To ensure consistent scoring across validators, we recommend all validators use the same LLM configuration. Contact the team for API access.

**Configuration:**

```bash
# In your .env file:
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
OPENAI_API_KEY=<contact team for API key>
```

**Auto-detection order:**
1. `OPENAI_API_BASE` + `OPENAI_API_KEY` → uses custom endpoint
2. `OPENAI_API_KEY` only → uses OpenAI
3. Local endpoints at `localhost:8000`, `localhost:30000`, `localhost:11434`

**Works with:** OpenAI, Zhipu GLM, vLLM, SGLang, Ollama, or any OpenAI-compatible endpoint.

**Without LLM:** Validator still works, narrative score defaults to 15/30. However, this may cause scoring differences with other validators.

---

## Installation

### Docker (Recommended)

```bash
git clone https://github.com/StorynetAI/storynet-subnet.git
cd storynet-subnet/Docker/validator
cp .env_example .env
nano .env  # Set WALLET_NAME, HOTKEY_NAME
docker compose up -d
```

### Manual

```bash
git clone https://github.com/StorynetAI/storynet-subnet.git
cd storynet-subnet
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python neurons/validator.py \
    --netuid 92 \
    --wallet.name <wallet> \
    --wallet.hotkey <hotkey> \
    --subtensor.network finney \
    --axon.port 19292
```

---

## Monitoring

```bash
btcli wallet overview --wallet.name <wallet>
btcli weights --netuid 92
```

---

## Support

- GitHub: https://github.com/StorynetAI/storynet-subnet/issues
- Discord: Bittensor Discord → #subnet-92
