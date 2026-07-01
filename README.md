# sec-agent

A cybersecurity AI agent combining LLMs and reinforcement learning for security research and education.

**This project is for learning and authorized testing only. Only point it at systems you own or that are explicitly designed for security testing (Juice Shop, DVWA, HackTheBox, TryHackMe, scanme.nmap.org).**

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Orchestrator                   │
│  ┌─────────────┐        ┌────────────────────┐  │
│  │ LLM Provider│        │   Tool Registry    │  │
│  │  (Groq /    │        │  ┌─────────────┐   │  │
│  │   Ollama)   │        │  │  HttpTool   │   │  │
│  └─────────────┘        │  │  NmapTool   │   │  │
│                         │  └─────────────┘   │  │
│                         └────────────────────┘  │
└─────────────────────────────────────────────────┘
                          │
              ┌───────────▼──────────┐
              │   Gymnasium Env      │
              │  (SecAgentEnv)       │
              └───────────┬──────────┘
                          │
              ┌───────────▼──────────┐
              │  RL Policy (SB3/PPO) │
              └──────────────────────┘
```

The RL policy selects a high-level action strategy (recon / http-probe / unconstrained). The LLM handles low-level reasoning and tool-call generation within that constraint. The tool layer executes real actions against allowlisted targets only.

---

## Quickstart

### Requirements

- Python 3.11+
- nmap installed and on PATH (`nmap -V` to verify)
- A free [Groq API key](https://console.groq.com) — or Ollama running locally

### Install

```bash
git clone https://github.com/yourusername/sec-agent
cd sec-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Configure

Edit `configs/default.yaml` to set your target hosts, model, and training parameters. The allowlists in `targets.http_hosts` and `targets.nmap_hosts` are the hard safety boundary — tools refuse to act outside them.

### Run the LLM-only agent

```bash
python experiments/06_test_orchestrator_both_tools.py
```

### Train the RL policy

```bash
python agent/rl/train.py
```

### Evaluate the trained policy

```bash
python agent/rl/evaluate.py
```

### Switch to Ollama (local, no API key)

1. Install [Ollama](https://ollama.com) and pull a model: `ollama pull llama3.2`
2. In `configs/default.yaml` set `llm.provider: ollama`
3. Run any script as normal — no API key needed

---

## Project structure

```
sec-agent/
├── agent/
│   ├── llm/
│   │   ├── base.py             # LLMProvider abstract class + LLMResponse
│   │   ├── factory.py          # builds the right provider from config
│   │   ├── groq_provider.py    # Groq (free API)
│   │   └── ollama_provider.py  # Ollama (local inference)
│   ├── tools/
│   │   ├── base.py             # Tool abstract class
│   │   ├── http_tool.py        # HTTP request tool
│   │   ├── nmap_tool.py        # Nmap scan tool
│   │   ├── registry.py         # Tool registration + dispatch
│   │   └── security.py         # Shared allowlist + validation helpers
│   ├── rl/
│   │   ├── env.py              # Gymnasium SecAgentEnv
│   │   ├── train.py            # SB3/PPO training loop
│   │   └── evaluate.py         # Policy evaluation + action distribution
│   ├── memory/
│   │   └── episode_log.py      # JSONL episode logging
│   ├── orchestrator.py         # LLM + tool loop (run / step_once)
│   └── config.py               # YAML config loader
├── configs/
│   └── default.yaml
├── tests/
│   └── test_smoke.py           # No API key needed — mocked LLM
├── experiments/                # One-off test scripts
├── docs/
├── .github/workflows/ci.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## Safety

- All tools enforce a host allowlist at the code level — prompts alone are never the safety boundary.
- Targets are validated for injection attempts before being passed to subprocesses.
- `subprocess` calls never use `shell=True`.
- The repo is intended for use only against systems you own or that explicitly permit testing.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
