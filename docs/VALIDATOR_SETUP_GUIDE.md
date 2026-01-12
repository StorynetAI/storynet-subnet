# StoryNet (SN92) Validator Setup Guide

> For Large Validators and Infrastructure Operators
> Last Updated: 2025-01-12

---

## Quick Info

| Item | Value |
|------|-------|
| **Subnet ID** | 92 |
| **Network** | Finney (Mainnet) |
| **GitHub Repo** | https://github.com/StorynetAI/storynet-subnet |
| **Protocol Version** | 3.2.1 |

---

## 1. Hardware Requirements

### Minimum Requirements

| Resource | Requirement |
|----------|-------------|
| **vCPU** | 2 cores |
| **RAM** | 4 GB |
| **Storage** | 20 GB SSD |
| **GPU** | Not required |
| **Network** | Stable internet, port 19292 open |

### Recommended Requirements

| Resource | Recommendation |
|----------|----------------|
| **vCPU** | 4 cores |
| **RAM** | 8 GB |
| **Storage** | 50 GB SSD |
| **GPU** | Not required |
| **Network** | Low latency connection |

**Note:** StoryNet validator does NOT require GPU. All computation is CPU-based scoring.

---

## 2. External API Services

### Required APIs

| Service | Purpose | Cost |
|---------|---------|------|
| **LLM API** (choose one) | Narrative quality scoring | Pay-per-use |

**Supported LLM Providers:**
- OpenAI (GPT-4)
- Zhipu AI (GLM-4)
- Google Gemini

### API Key Setup

```bash
# Option 1: OpenAI
export OPENAI_API_KEY=sk-your-key-here

# Option 2: Zhipu AI (recommended for cost)
export ZHIPU_API_KEY=your-zhipu-key-here

# Option 3: Gemini
export GEMINI_API_KEY=your-gemini-key-here
```

**Estimated API Cost:** ~$10-30/month depending on validation frequency

---

## 3. Installation Methods

### Method A: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/StorynetAI/storynet-subnet.git
cd storynet-subnet/Docker/validator

# Configure environment
cp .env_example .env
nano .env  # Set WALLET_NAME, HOTKEY_NAME, and API key

# Deploy with auto-updates
docker compose up -d

# Check logs
docker compose logs -f
```

### Method B: Manual Installation

```bash
# Clone repository
git clone https://github.com/StorynetAI/storynet-subnet.git
cd storynet-subnet

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY=sk-xxx  # or ZHIPU_API_KEY

# Run validator
python neurons/validator.py \
    --netuid 92 \
    --wallet.name <your_wallet> \
    --wallet.hotkey <your_hotkey> \
    --subtensor.network finney \
    --axon.port 19292 \
    --logging.info
```

---

## 4. Validation Process

### What the Validator Does

1. **Queries miners** with story generation tasks (blueprint, characters, story arc, chapters)
2. **Scores responses** using a 100-point multi-dimensional system:
   - Technical (20 pts): JSON validity, schema compliance, response time
   - Structure (30 pts): Story coherence, chapter progression
   - Content (20 pts): Relevance, fluency, originality
   - Narrative (30 pts): AI-evaluated creative quality (requires LLM API)
3. **Sets weights** on-chain based on miner scores

### Validation Frequency

- Default: Every ~20 minutes (tempo-based)
- Configurable via `--neuron.epoch_length`

---

## 5. Monitoring

```bash
# Check validator status
btcli wallet overview --wallet.name <wallet>

# Check subnet weights
btcli weights --netuid 92

# Docker logs
docker compose logs -f sn92-validator
```

---

## 6. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| API key not working | Verify environment variable is set correctly |
| Connection timeout | Check firewall, ensure port 19292 is open |
| Low VTrust | Ensure validator is running and setting weights regularly |

### Support

- Discord: Bittensor Discord → #subnet-92
- GitHub Issues: https://github.com/StorynetAI/storynet-subnet/issues

---

## 7. FAQ

**Q: Does validator need GPU?**
A: No, StoryNet validator is CPU-only.

**Q: What's the expected ROI?**
A: Depends on stake and subnet emissions. Check taostats.io for current rates.

**Q: How often should I update?**
A: Docker with Watchtower auto-updates. Manual installs should check for updates weekly.

---

## Contact

For validator coordination or questions:
- GitHub: https://github.com/StorynetAI/storynet-subnet
- Discord: Bittensor Discord → #subnet-92
