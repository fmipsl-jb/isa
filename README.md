# Intelligent Search Assistant

The Intelligent Search Assistant (ISA) is a Streamlit workbench for exploring OpenAI's Responses API with routing, prompt management, and inspection tools aimed at product and support teams. The app mirrors a production-style integration while keeping the code path compact enough to iterate locally.

---

## Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
  - [Streamlit secrets](#streamlit-secrets)
  - [Environment variables](#environment-variables)
  - [Prompt defaults](#prompt-defaults)
- [Running the app](#running-the-app)
- [Using the interface](#using-the-interface)
  - [Conversation flow](#conversation-flow)
  - [Routing behaviour](#routing-behaviour)
  - [Streaming and responses](#streaming-and-responses)
  - [Tools and metadata](#tools-and-metadata)
- [Customization guide](#customization-guide)
- [Troubleshooting](#troubleshooting)
- [Development notes](#development-notes)

---

## Overview
ISA provides an interactive console for evaluating how different prompts and models respond to user questions. It loads optional developer and user prompts, classifies each query to choose the best route, and exposes both rendered answers and the underlying JSON response. The intent is to accelerate prompt iteration while maintaining visibility into the request payloads a production system would send.

## Features
- **Automatic routing** – A lightweight classifier evaluates each question and selects a deterministic or creative route before contacting the main model.
- **Prompt asset support** – Optional integration with the OpenAI Prompts API keeps default text in sync with centrally managed prompt assets.
- **Streaming viewer** – Models that support streaming display token-level updates; synchronous models appear once complete.
- **Tool integration** – Requests can include file search and web search tools, making it easy to experiment with retrieval-augmented responses.
- **Raw payload inspector** – Every run stores the full JSON payload in an expander so you can review usage, tool calls, and metadata.
- **Session awareness** – ISA preserves conversation state per model so follow-up questions stay in context when supported by the API.

## Architecture
The application is a single Streamlit page defined in `app.py`:
1. **Client bootstrap** – `build_client()` resolves the OpenAI API key from Streamlit secrets or the `OPENAI_API_KEY` environment variable.
2. **Prompt loading** – `load_default_user_prompt()` and `load_developer_prompt()` pull optional defaults from secrets or the Prompts API.
3. **Classification** – `classify_user_prompt()` runs a prompt-based classifier to determine the route and metadata for the upcoming request.
4. **Run configuration** – `build_route_run_config()` selects the target model, decoding parameters, and prompt reference for each route.
5. **Execution** – `run_model()` sends the Responses API request, handles streaming callbacks, and normalizes the output text.
6. **UI rendering** – Streamlit widgets display the request form, track active conversations, and surface raw responses for inspection.

There are no background services or databases; all state lives in Streamlit's session state.

## Prerequisites
- Python **3.9 or newer**.
- An OpenAI account with access to the Responses API and the models you intend to test.
- Network access to `api.openai.com`.

## Setup
Clone the repository and install dependencies. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
ISA reads configuration from Streamlit's secrets and from environment variables. Populate only the values required for your workflows—avoid storing sensitive keys directly in the repository.

### Streamlit secrets
Create a `.streamlit/secrets.toml` file alongside `app.py`. At minimum provide your OpenAI key and any prompt metadata you plan to use:

```toml
[openai]
api_key = "YOUR_OPENAI_KEY"

[prompts]
# Optional: default_user_prompt_id = "prompt-id-from-openai"
# Optional: developer_prompt = "Initial system/developer instructions"
# Optional: prompt_app.id = "prompt-asset-id"
# Optional: prompt_creative.id = "another-prompt-asset-id"
```

Only store placeholder values in version control. Real keys should be added in your deployment environment or ignored via `.gitignore`.

### Environment variables
If Streamlit secrets are not available, set `OPENAI_API_KEY` before launching the app:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

Environment variables can also be useful for local development when you prefer not to create a secrets file.

### Prompt defaults
Prompt configurations under `[prompts]` allow you to associate prompt asset IDs, cache keys, variable mappings, and other metadata with the app. ISA will pass the active user question into any configured variables. Refer to `get_prompt_config()` in `app.py` for the supported fields.

## Running the app
Launch Streamlit from the repository root:

```bash
streamlit run app.py
```

Streamlit will output a local URL (typically `http://localhost:8501`). Keep the terminal open while using the interface. When deploying to Streamlit Community Cloud or another hosting provider, ensure your secrets and environment variables are configured in that environment.

## Using the interface
### Conversation flow
1. Enter a question in the **User question** text area. If a default user prompt is configured, it will appear automatically.
2. Select your digital audio workstation (DAW) version from the dropdown. ISA stores the selection with the conversation so follow-up messages stay consistent.
3. Click **Generate responses** to submit the query.
4. Review the assistant reply and open the **Show raw response** expander to inspect the full payload.

### Routing behaviour
- The classifier prompt assigns a token such as `APP` or `CREATIVE`. ISA maps that token to one of two routes.
- Route 1 targets a deterministic model (default: `gpt-4.1-nano`) with streaming enabled.
- Route 2 targets a higher-capability model (default: `gpt-5-nano`) with reasoning controls and synchronous delivery.
- Conversations stick to the route and model chosen for the first message unless you manually reset the session state.

### Streaming and responses
- Streaming responses update the message area in real time using `client.responses.stream`.
- Non-streaming responses display only after the API call finishes.
- All responses are stored (`store=True`) so they appear in the OpenAI console history unless you adjust the request parameters.

### Tools and metadata
- Requests may include a file search tool and, for creative routes, a domain-scoped web search tool. Replace the placeholder tool configuration in `run_model()` with your own resource identifiers before deploying to production.
- ISA attaches metadata such as the classifier token, detected language, and selected DAW version to each request. Adjust `resolve_metadata_token()` and related helpers to suit your own taxonomy.

## Customization guide
- **Models and parameters** – Edit `DEFAULT_MODELS` and `build_route_run_config()` to change available models, temperatures, or reasoning settings.
- **Prompt references** – Update prompt names or variable mappings in `.streamlit/secrets.toml` to align with your managed prompt assets.
- **Tooling** – Modify the `tools` list in `run_model()` to enable/disable retrieval or function tools, or to inject custom tool definitions.
- **UI changes** – Streamlit components are defined in `main()`. You can reintroduce a sidebar, add tabs, or extend the layout for additional diagnostics.

## Troubleshooting
| Issue | Suggested action |
| --- | --- |
| "OpenAI API key not found" | Ensure the key is set in `.streamlit/secrets.toml` or exported as `OPENAI_API_KEY` before running Streamlit. |
| Default prompt fails to load | Confirm the prompt ID exists and that the API key has Prompts API access; ISA will fall back to an empty text area. |
| Classifier never routes to the creative model | Inspect your classifier prompt output and adjust token mapping in `determine_route()`. |
| Streaming stops unexpectedly | Review the raw payload for `response.error` events and consider disabling streaming for the affected model. |
| Web search results missing | Verify that your deployment account has access to the web search tool and update the tool configuration with valid identifiers. |

## Development notes
- The page configuration is set via `st.set_page_config(page_title="Intelligent Search Assistant", layout="wide")`.
- Markdown rendering is handled by Streamlit; sanitize or transform model outputs if you expect HTML responses.
- Conversation state, including the active model and metadata, is stored in `st.session_state` under the `conversations` key.
- Error handling around streaming is intentionally defensive to avoid leaving the UI in an inconsistent state during experiments.
- Contributions are welcome via pull request. Please avoid committing secrets or proprietary identifiers.
