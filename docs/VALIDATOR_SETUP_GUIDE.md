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

## External Dependencies

**None required.** All scoring is rule-based and deterministic.

No GPU, no LLM API, no external services needed.

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
