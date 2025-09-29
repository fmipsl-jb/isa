# Intelligent Search Assistant (ISA)

The Intelligent Search Assistant is a Streamlit-based workbench for experimenting with the OpenAI Responses API. Version 2 adds streaming output, built-in tool calls, and first-class support for prompt assets so that teams can evaluate models and prompts in a workflow that mirrors production integrations. This README is intended to give a developer everything they need to understand how the app works, how to configure it, and how to adapt it for real-world deployments.

---

## Table of Contents
- [High-Level Overview](#high-level-overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Secrets file](#secrets-file)
  - [Environment variables](#environment-variables)
  - [Optional prompt defaults](#optional-prompt-defaults)
- [Running the Application](#running-the-application)
  - [Local development](#local-development)
  - [Streamlit Community Cloud](#streamlit-community-cloud)
- [Using ISA](#using-isa)
  - [Main workspace](#main-workspace)
  - [Automatic routing](#automatic-routing)
  - [Sidebar configuration (hidden in Version 3)](#sidebar-configuration-hidden-in-version-3)
  - [Real-time streaming](#real-time-streaming)
  - [Inspecting raw responses](#inspecting-raw-responses)
- [How the Response Pipeline Works](#how-the-response-pipeline-works)
  - [Message construction](#message-construction)
  - [Model-specific controls](#model-specific-controls)
  - [Tool invocation](#tool-invocation)
  - [Streaming vs. non-streaming execution](#streaming-vs-non-streaming-execution)
- [Customization Guide](#customization-guide)
  - [Supported models](#supported-models)
  - [Adjusting tool usage](#adjusting-tool-usage)
  - [Adapting the UI](#adapting-the-ui)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)
- [Future Enhancements](#future-enhancements)

---

## High-Level Overview

ISA is a diagnostic interface for OpenAI's Responses API. It enables product and ML teams to:

- Automatically route each question between deterministic application handling (Route 1) and a creative path (Route 2) based on a background classifier.
- Prototype prompt changes quickly with automatic retrieval of saved prompt assets.
- Observe the raw JSON payloads that would be returned to a production caller, including tool call metadata.
- Share reproducible experiments with other stakeholders (Streamlit retains control state between reruns).

The application is intentionally thin—it does not persist conversations or user data—but it mirrors how a production system would construct requests, manage secrets, and toggle model-specific options.

## System Architecture

ISA is a single-page Streamlit application built around the `app.py` entry point. Key components include:

1. **Client initialization** – `build_client()` resolves an API key from Streamlit secrets or the `OPENAI_API_KEY` environment variable and instantiates the `OpenAI` client.
2. **Classifier routing** – `classify_user_prompt()` selects the route (and therefore model/parameters) based on a managed prompt asset.
3. **Prompt preparation** – `load_default_user_prompt()` optionally fetches a saved prompt from the OpenAI Prompts API, while `load_developer_prompt()` loads a developer/system message from secrets.
4. **Execution engine** – `run_model()` constructs the request payload, handles streaming callbacks, and normalizes the returned text and JSON for presentation.
5. **UI rendering** – Streamlit widgets display response text and provide an expandable JSON inspector per model.

This layout keeps business logic in `app.py` while leveraging Streamlit for layout and state management. There are no additional microservices or background workers.

## Key Features

- **Automatic routing** – A background classifier selects Route 1 (gpt-4.1-nano with streaming) for deterministic application responses or Route 2 (gpt-5-nano without streaming) for creative needs.
- **Managed prompt defaults** – Automatically hydrate the main textarea with content retrieved from the latest version of a managed prompt asset.
- **Prompt-level caching and storage** – All requests send `store=True` and reuse prompt cache keys so calls show up in the console history while benefiting from caching.
- **Integrated agent tools** – Automatically attach the OpenAI `file_search` tool for grounded answers on every route and enable a scoped `web_search` tool for creative responses.
- **Streaming output where supported** – Route 1 streams token deltas live in the UI, while Route 2 completes synchronously for GPT-5 models.
- **Raw response explorer** – Inspect metadata such as latency, usage, and tool call transcripts inside an expandable panel.
- **Graceful validation** – Inline warnings prevent empty prompts or missing API keys from triggering API requests.

## Prerequisites

- Python **3.9+** (Streamlit Cloud currently ships Python 3.9; Python 3.10 and 3.11 also work locally).
- An OpenAI account with access to the **Responses API** and the models you intend to test (`gpt-4.1`, `gpt-4o`, `gpt-5`, etc.).
- Network access to `api.openai.com` from the environment running the app.

## Installation

1. Clone or download this repository.
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

ISA expects configuration to live in Streamlit secrets or environment variables. The Streamlit secrets approach is preferred because it also works seamlessly on Streamlit Community Cloud.

### Secrets file

Create `.streamlit/secrets.toml` at the project root (see `.streamlit/secrets.example.toml` for a complete template):

```toml
[openai]
api_key = "sk-your-api-key"

[prompts.prompt_app]
id = "prompt_app_id"
variable_names = ["user_input"]
cache_key = "isa-app"

[prompts.prompt_creative]
id = "prompt_creative_id"
variable_names = ["user_input"]
cache_key = "isa-creative"

[prompts.prompt_classifier]
id = "prompt_classifier_id"
variable_names = ["user_input"]
model = "gpt-4.1-mini"
cache_key = "isa-classifier"
```

### Environment variables

If you cannot use Streamlit secrets, export environment variables before launching Streamlit:

```bash
export OPENAI_API_KEY="sk-your-api-key"
```

Environment variables take precedence only when the associated value is not present in Streamlit secrets. You can easily modify `load_default_user_prompt()` to check for additional environment-based overrides if needed.

### Optional prompt defaults

Optionally, you can add a `[prompts]` block to control how the main textarea and the developer instructions are pre-populated:

- `developer_prompt` – Injected as a developer message (`role="developer"`) before each user prompt. Use this to enforce tone, format, or tool instructions.
- `default_user_prompt_id` – Identifier of a saved prompt asset in the OpenAI Prompts API. When set, ISA fetches the prompt during page load.

The nested tables `[prompts.prompt_app]`, `[prompts.prompt_creative]`, and `[prompts.prompt_classifier]` configure the prompt assets and cache keys used for each agent. Provide the prompt IDs and variable names expected by every prompt. Cache keys are optional but recommended to keep call history organized.

If the Prompts API call fails or returns no textual content, ISA surfaces a warning and the textarea falls back to an empty string.

## Running the Application

### Local development

Launch Streamlit from the repository root:

```bash
streamlit run app.py
```

Streamlit will display a local URL (typically `http://localhost:8501`). Keep the terminal session open while using the app.

### Streamlit Community Cloud

1. Push this repository to GitHub (or another Git provider supported by Streamlit).
2. In Streamlit Cloud, create a new app targeting `app.py`.
3. In **App settings → Secrets**, paste the content of your `.streamlit/secrets.toml`.
4. Deploy. Streamlit Cloud will install dependencies from `requirements.txt` and boot the app automatically.

## Using ISA

### Main workspace

- **User question** – Primary textarea populated with the default prompt (if configured). Accepts Markdown; trailing whitespace is stripped before sending to the API.
- **Generate responses** – Primary action button. Validation ensures a prompt is provided before triggering the background classifier and downstream Responses API call.
- **Response area** – Shows the ongoing conversation for the model selected by the classifier together with the latest assistant reply.

### Automatic routing

- ISA calls a background prompt (`prompt_classifier`) whose JSON output determines the route. Tokens of `APP` or `OOS` select **Route 1** (gpt-4.1-nano with streaming). Tokens of `CREATIVE` or `HYBRID` select **Route 2** (gpt-5-nano without streaming).
- The classifier call is silent to end users. It always sends `store=True`, reuses a cache key, and passes the user question through prompt variables defined in secrets.
- Route 1 uses `prompt_app`, `temperature=0.5`, `top_p=0.8`, and streaming updates. Route 2 uses `prompt_creative` with `reasoning.effort="low"`, `text.verbosity="low"`, and synchronous execution.

### Sidebar configuration (removed in Version 3)

Version 3 removes the configuration sidebar entirely. The previous sliders and multi-select have been deleted from the codebase to prevent residual errors tied to the old implementation.

### Real-time streaming

Route 1 streams responses from `gpt-4.1-nano`. As OpenAI emits `response.output_text.delta` events, the app updates the assistant message live. Route 2 (gpt-5-nano) runs synchronously because GPT-5 models do not support streaming.

### Inspecting raw responses

Each response includes a **“Show raw response”** expander containing the full JSON payload returned by the Responses API. Use this to inspect tool invocations, usage metrics, reasoning traces, or content safety fields.

## How the Response Pipeline Works

### Message construction

1. ISA builds a list of messages starting with the optional developer prompt (`role="developer"`) followed by the user prompt (`role="user"`).
2. Every message is encoded using the `input_text` content type to mirror the canonical Responses API format.
3. The request passes `store=True` so the interaction appears in the OpenAI console history.

### Model-specific controls

- Route 1 uses `gpt-4.1-nano` with `temperature=0.5` and `top_p=0.8`.
- Route 2 uses `gpt-5-nano` with `reasoning.effort="low"` and `text.verbosity="low"`. GPT-5 models do not accept `temperature`/`top_p`, so those parameters are omitted automatically.

### Tool invocation

ISA attaches a `file_search` tool definition to both routes and layers on a scoped `web_search` tool (limited to PreSonus domains) for the creative agent. The file search tool references a pre-built vector store (`vs_68c92dcc842c81919b9996ec34b55c2c`), and the creative route requests `include=["web_search_call.action.sources"]` so raw responses surface citation metadata.

To disable or customize tool usage, edit the `tools` list construction in `run_model()` or make the vector store ID configurable via secrets.

### Streaming vs. non-streaming execution

- Route 1 invokes `run_model(..., stream=True)` so the UI can display deltas while the response is being generated.
- Route 2 invokes `run_model(..., stream=False)` and renders the full answer only after completion because GPT-5 models do not support streaming.

## Customization Guide

### Supported models

Route 1 is hard-coded to `gpt-4.1-nano` and Route 2 to `gpt-5-nano`. To adjust the routing targets or decoding parameters, edit `build_route_run_config()` in `app.py`.

### Adjusting tool usage

- **Different vector store** – Replace the hard-coded vector store ID with one sourced from secrets.
- **Additional tools** – Append tool definitions (e.g., function calling) inside the `params` dict in `run_model()`.
- **Disabling tools** – Remove entries from the `tools` list inside `run_model()` or gate them behind new configuration toggles before the request is sent.

### Adapting the UI

- **Custom prompt panes** – Wrap the textarea and response columns in `st.tabs()` for multi-scenario testing.
- **Session persistence** – Use Streamlit session state to keep a history of prompts/responses.
- **Localization** – The placeholder text already demonstrates multilingual prompts. To localize validation messages, edit the strings inside `main()` where warnings and errors are raised.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| "OpenAI API key not found" | Ensure the API key is present in `.streamlit/secrets.toml` or exported as `OPENAI_API_KEY` before launching the app. |
| "Please enter a question" warning | The prompt textarea is empty. Enter text and click **Generate responses** again. |
| APIConnectionError or APIError | These are upstream errors from OpenAI. Retry, verify network access, and confirm the model is enabled for your account. |
| Default prompt fails to load | Confirm the prompt ID exists and that the API key has access to the Prompts API. ISA will warn and fall back to an empty textarea. |
| Unexpected streaming error | Streaming raises `response.error` events when the server aborts the stream. Review the raw JSON to diagnose; fall back to non-streaming execution if necessary. |

## Development Notes

- The Streamlit page is configured via `st.set_page_config(page_title="Intelligent Search Assistant", layout="wide")`.
- Responses are rendered as Markdown. If models output HTML, sanitize or adjust the renderer before displaying.
- ISA stores all responses in the OpenAI console (`store=True`). Disable this if your security policy disallows storage.
- The developer prompt loader intentionally returns `None` when the secret is absent; `build_input_messages()` only sends the developer role when content is provided.
- Error handling intentionally catches broad exceptions around the streaming handler to prevent the UI from locking up. Replace with more granular logging if needed.
