# DM Forge

DM Forge generates human-sounding LinkedIn DMs from:
- raw LinkedIn post text
- a public LinkedIn profile URL
- auto-inferred context from the post or URL

The app does **not** send LinkedIn messages automatically. It only creates draft messages for manual use.

---

## What It Does

1. **Input adapter** resolves post text from pasted content or a LinkedIn URL.
2. **Research agent** analyzes the post and decides what kind of response fits best.
3. **Strategy agent** plans whether to answer, acknowledge, relate, ask a follow-up, or share a perspective.
4. **Writer agent** drafts the DM based on that plan.
5. **Editor agent** reviews quality, fixes issues if needed, and returns the final message.
6. **UI** shows the final DM in a Gradio web app.

The LLM decides the response strategy dynamically. For example:
- If the post asks a question, the DM should answer it first.
- If the post shares an achievement, the DM should acknowledge it specifically.
- If the post shares knowledge, the DM can ask a thoughtful follow-up.

All LLM calls go through the **LLM router** (`llm/router.py`), which handles provider failover, caching, and optional offline fallback.

---

## Prerequisites

- **Python 3.10+**
- Internet access for LLM API calls
- At least **one** working LLM provider, unless you use demo mode or offline fallback

---

## Step-by-Step Setup

### 1. Go to the project folder

```bash
cd /path/to/dm-forge
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```bash
.venv\Scripts\activate
```

**Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt.

### 4. Upgrade pip (recommended)

```bash
pip install --upgrade pip
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `gradio` — web UI
- `crewai` — agent pipeline
- `openai` — Groq/OpenRouter client
- `requests`, `beautifulsoup4` — HTTP and HTML parsing
- `crawl4ai` — optional LinkedIn URL crawling

### 6. Create your environment file

```bash
cp .env.example .env
```

Then open `.env` and add at least one API key.

Example:

```env
GROQ_API_KEY=your_groq_key_here
DM_FORGE_PROVIDERS=groq,openrouter,gemini,ollama
DM_FORGE_USE_CRAWL4AI_FOR_LINKEDIN=0
```

The app loads `.env` automatically when you run `main.py`.

---

## Run the Application

Make sure your virtual environment is activated, then run:

```bash
python main.py
```

This starts the Gradio web app. By default, open:

**http://127.0.0.1:7860**

### How to use the UI

1. Paste a **LinkedIn post** or a **public profile URL**
2. Click **Generate DM**
3. Copy the generated message and send it manually on LinkedIn

Example inputs:
- Post text: `After 9 months of customer interviews, we launched a no-code analytics add-on.`
- Profile URL: `https://www.linkedin.com/in/username/`

---

## Demo Mode (No API Keys Required)

To test the pipeline without live LLM calls:

```bash
python -m demo.run_demo
```

This seeds the cache with predefined outputs and runs sample posts in the terminal.

---

## LLM Provider Setup

The router tries providers in the order defined by `DM_FORGE_PROVIDERS`.

Supported providers:
- **Groq**
- **OpenRouter**
- **Gemini**
- **Ollama** (local)

### Option 1: Use `.env`

```env
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
GEMINI_API_KEY=your_key

GROQ_MODEL=llama-3.3-70b-versatile
OPENROUTER_MODEL=openai/gpt-4o-mini
GEMINI_MODEL=gemini-1.5-flash
OLLAMA_MODEL=llama3.2

DM_FORGE_PROVIDERS=groq,openrouter,gemini,ollama
DM_FORGE_TIMEOUT_SECONDS=30
DM_FORGE_ALLOW_OFFLINE_FALLBACK=0
OLLAMA_URL=http://localhost:11434/api/generate
```

### Option 2: Export environment variables

```bash
export GROQ_API_KEY="your_key_here"
export DM_FORGE_PROVIDERS="groq,openrouter,gemini,ollama"
```

### Use Ollama locally

1. Install and start Ollama
2. Pull a model:

```bash
ollama pull llama3.2
```

3. Make sure Ollama is running at `http://localhost:11434`

### Optional settings

| Variable | Purpose |
|---|---|
| `DM_FORGE_DEFAULT_SENDER_PROFILE` | Reusable sender context for all runs |
| `DM_FORGE_USE_CRAWL4AI_FOR_LINKEDIN=0` | Recommended. Uses lightweight LinkedIn URL extraction and avoids crawl4ai anti-bot issues |
| `DM_FORGE_ALLOW_OFFLINE_FALLBACK=1` | Uses a local fallback response if all providers fail |

---

## Folder Structure

```text
dm-forge/
├── main.py              # App entry point
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── llm/                 # Model routing, provider failover, SQLite cache
├── crew/                # Research, strategy, writer, editor pipeline
├── input/               # Input adapter and URL crawling
├── ui/                  # Gradio interface
└── demo/                # Offline demo fixtures
```

---

## Troubleshooting

### `All providers failed`
- Add at least one valid API key in `.env`
- Or start Ollama locally
- Or set `DM_FORGE_ALLOW_OFFLINE_FALLBACK=1`

### LinkedIn URL returns little or no text
- LinkedIn often blocks scraping
- Paste the post text directly instead of the URL
- Keep `DM_FORGE_USE_CRAWL4AI_FOR_LINKEDIN=0`

### `Please provide a LinkedIn post or profile URL`
- The input box was empty

### Port already in use
- Another app may already be using port `7860`
- Stop the other app and run again

### Virtual environment not active
- Run `source .venv/bin/activate` before `python main.py`

---

## Disclaimer

This tool only uses public information and generates draft messages. It does not send LinkedIn messages automatically. Always review and personalize messages before sending.
