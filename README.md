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
  - [Sidebar configuration](#sidebar-configuration)
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

- Compare responses from up to two models side by side using identical prompts.
- Prototype prompt changes quickly with automatic retrieval of saved prompt assets.
- Observe the raw JSON payloads that would be returned to a production caller, including tool call metadata.
- Share reproducible experiments with other stakeholders (Streamlit retains control state between reruns).

The application is intentionally thin—it does not persist conversations or user data—but it mirrors how a production system would construct requests, manage secrets, and toggle model-specific options.

## System Architecture

ISA is a single-page Streamlit application built around the `app.py` entry point. Key components include:

1. **Client initialization** – `build_client()` resolves an API key from Streamlit secrets or the `OPENAI_API_KEY` environment variable and instantiates the `OpenAI` client.
2. **Sidebar configuration** – `render_sidebar()` collects the user's model selections and decoding parameters.
3. **Prompt preparation** – `load_default_user_prompt()` optionally fetches a saved prompt from the OpenAI Prompts API, while `load_developer_prompt()` loads a developer/system message from secrets.
4. **Execution engine** – `run_model()` constructs the request payload, handles streaming callbacks, and normalizes the returned text and JSON for presentation.
5. **UI rendering** – Streamlit widgets display response text and provide an expandable JSON inspector per model.

This layout keeps business logic in `app.py` while leveraging Streamlit for layout and state management. There are no additional microservices or background workers.

## Key Features

- **Versioned prompt defaults** – Automatically hydrate the main textarea with content retrieved from a managed prompt asset.
- **Side-by-side model comparisons** – Choose any two models from a curated list to evaluate response differences in real time.
- **Granular decoding controls** – Toggle temperature/top-p for GPT-4.x models and reasoning effort/verbosity for GPT-5-family models.
- **Integrated file search tool** – Automatically attach the OpenAI `file_search` tool (with a predefined vector store) for grounded answers when reasoning effort is above `minimal`.
- **Streaming output** – Watch token deltas update live in the UI while the full JSON payload is collected once streaming completes.
- **Raw response explorer** – Inspect metadata such as latency, usage, and tool call transcripts inside an expandable panel.
- **Graceful validation** – Inline warnings prevent empty prompts, missing API keys, or missing model selections from triggering API requests.

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

Create `.streamlit/secrets.toml` at the project root:

```toml
[openai]
api_key = "sk-your-api-key"

[prompts]
# Optional entries – see below for details.
# developer_prompt = "Act as a friendly mentor..."
```

### Environment variables

If you cannot use Streamlit secrets, export environment variables before launching Streamlit:

```bash
export OPENAI_API_KEY="sk-your-api-key"
# Optional: override the prompt ID or version without editing secrets.
```

Environment variables take precedence only when the associated value is not present in Streamlit secrets. `ISA_DEFAULT_PROMPT_ID` and `ISA_DEFAULT_PROMPT_VERSION` are not read directly by the app, but you can easily modify `load_default_user_prompt()` to check for them if you prefer environment-based configuration.

### Optional prompt defaults

The `[prompts]` block controls how the main textarea and the developer instructions are pre-populated:

- `developer_prompt` – Injected as a developer message (`role="developer"`) before each user prompt. Use this to enforce tone, format, or tool instructions.
- `default_user_prompt_id` – Identifier of a saved prompt asset in the OpenAI Prompts API. When set, ISA fetches the prompt during page load.
- `default_user_prompt_version` – (Default `6`) Which version of the prompt asset to retrieve. If omitted or blank, ISA falls back to version `8`.

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
- **Generate responses** – Primary action button. Validation ensures a prompt and at least one model are selected before proceeding.
- **Response columns** – The app creates one column per selected model (maximum two) and renders Markdown output in each column.

### Sidebar configuration

- **Models to query** – Multiselect list based on `DEFAULT_MODELS`. Defaults to the first entry only to encourage single-model tests unless a comparison is needed.
- **Temperature** – Enabled for non GPT-5 models. Higher values increase randomness.
- **Top P** – Nucleus sampling cutoff for non GPT-5 models. Values outside `[0, 1]` are clamped by the slider.
- **Reasoning effort** – Maps to `reasoning.effort` for GPT-5 models. Selecting `minimal` disables the file search tool to emphasize concise answers.
- **Response verbosity** – Maps to `text.verbosity` for GPT-5 models. Non GPT-5 models ignore this setting.

### Real-time streaming

ISA requests streaming responses by default. As OpenAI emits `response.output_text.delta` events, the app updates each column live. Once the stream finishes, the final consolidated text is displayed. If streaming fails for any reason, the app raises an inline error and skips rendering.

### Inspecting raw responses

Each model column includes a **“Show raw response”** expander containing the full JSON payload returned by the Responses API. Use this to inspect tool invocations, usage metrics, reasoning traces, or content safety fields.

## How the Response Pipeline Works

### Message construction

1. ISA builds a list of messages starting with the optional developer prompt (`role="developer"`) followed by the user prompt (`role="user"`).
2. Every message is encoded using the `input_text` content type to mirror the canonical Responses API format.
3. The request passes `store=True` so the interaction appears in the OpenAI console history.

### Model-specific controls

- For GPT-5-family models (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`), ISA sends `reasoning.effort` and `text.verbosity` options when applicable. Temperature and top-p are intentionally omitted because those models prioritize reasoning controls.
- For GPT-4.x models and other non GPT-5 entries, ISA applies `temperature` and (optionally) `top_p`. The slider values are sent only when supported by the target model.

### Tool invocation

Unless the user selects `reasoning effort = minimal`, ISA attaches a `file_search` tool definition to the request. The tool references a pre-built vector store (`vs_68c92dcc842c81919b9996ec34b55c2c`) and enables `include=["web_search_call.action.sources"]` to surface citation metadata in the raw JSON.

To disable tool usage globally, adjust the `use_file_search_tool` branch in `run_model()` or make the vector store ID configurable via secrets.

### Streaming vs. non-streaming execution

- When `stream=True`, ISA collects text deltas through the event stream while also compiling the final response object. The helper `extract_delta_text()` normalizes the event payload regardless of shape (string, dict, or object).
- If you want synchronous execution, set `stream=False` when calling `run_model()` and remove the callback logic. ISA will then call `client.responses.create(...)` and display the final text once the API call returns.

## Customization Guide

### Supported models

`DEFAULT_MODELS` lives at the top of `app.py`:

```python
DEFAULT_MODELS = [
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]
```

Modify this list to expose additional models. The sidebar limits selections to two to maintain a readable layout. If you need more than two columns, increase `max_selections` and update the layout accordingly.

### Adjusting tool usage

- **Different vector store** – Replace the hard-coded vector store ID with one sourced from secrets.
- **Additional tools** – Append tool definitions (e.g., function calling) inside the `params` dict in `run_model()`.
- **Disabling tools** – Set `use_file_search_tool = False` or add a new sidebar toggle that feeds into the branch that attaches the `tools` key.

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
| Default prompt fails to load | Confirm the prompt ID/version exist and that the API key has access to the Prompts API. ISA will warn and fall back to an empty textarea. |
| Unexpected streaming error | Streaming raises `response.error` events when the server aborts the stream. Review the raw JSON to diagnose; fall back to non-streaming execution if necessary. |

## Development Notes

- The Streamlit page is configured via `st.set_page_config(page_title="Intelligent Search Assistant", layout="wide")`.
- Responses are rendered as Markdown. If models output HTML, sanitize or adjust the renderer before displaying.
- ISA stores all responses in the OpenAI console (`store=True`). Disable this if your security policy disallows storage.
- The developer prompt loader intentionally returns `None` when the secret is absent; `build_input_messages()` only sends the developer role when content is provided.
- Error handling intentionally catches broad exceptions around the streaming handler to prevent the UI from locking up. Replace with more granular logging if needed.

## Future Enhancements

The following ideas were considered but are not implemented yet:

- Let users upload files that are automatically added to a transient vector store for retrieval.
- Persist comparison sessions so teams can revisit historical runs.
- Add guardrails around model availability (e.g., hide GPT-5 options when the key lacks access).
- Surface usage cost estimates alongside each response column.

Contributions are welcome—fork the repository, create a feature branch, and open a pull request with your improvements.
