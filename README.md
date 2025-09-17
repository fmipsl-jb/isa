# Intelligent Search Assistant (ISA)

The Intelligent Search Assistant is a Streamlit application for experimenting with the OpenAI Responses API. It lets you craft prompts, compare responses from up to two models side by side, and inspect the raw JSON returned by the API. The tool is ideal for prompt engineering sessions, regression testing prompt changes, or showcasing the behavior of different OpenAI models with the same user question.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuring Secrets](#configuring-secrets)
- [Running the App](#running-the-app)
  - [Local execution](#local-execution)
  - [Streamlit Community Cloud](#streamlit-community-cloud)
- [Using the Interface](#using-the-interface)
  - [Main panel](#main-panel)
  - [Sidebar controls](#sidebar-controls)
  - [Viewing raw responses](#viewing-raw-responses)
- [Customization](#customization)
  - [Default prompts](#default-prompts)
  - [Supported models](#supported-models)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

---

## Features

- **Side-by-side model comparison** – Select up to two models and view their responses to the same prompt in separate columns.
- **Prompt and developer instructions** – Combine a user prompt with an optional developer persona defined in Streamlit secrets.
- **Granular response tuning** – Adjust temperature, nucleus sampling (top_p), reasoning effort, and verbosity directly from the sidebar.
- **Raw JSON viewer** – Inspect the full Responses API payload in an expandable panel for each model invocation.
- **Graceful error handling** – In-app messages highlight missing API keys, empty prompts, or upstream API issues.

## Requirements

- Python 3.9 or newer (matching the version used by Streamlit Cloud)
- Access to the OpenAI Responses API with models such as `gpt-4.1-mini` or `gpt-4o`
- An OpenAI API key with sufficient quota for the models you intend to evaluate

## Quick Start

1. **Clone the repository** (or download the source).
2. *(Optional but recommended)* **Create a virtual environment.**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies.**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Configure your OpenAI API key** following [Configuring Secrets](#configuring-secrets).
5. **Run the Streamlit app** using the instructions in [Running the App](#running-the-app).

## Configuring Secrets

The app reads configuration from Streamlit secrets. Provide your API key in one of two ways:

1. **Secrets file (recommended)**
   - Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`.
   - Add your key under the `[openai]` section:
     ```toml
     [openai]
     api_key = "sk-your-real-key"
     ```

2. **Environment variable**
   - Export your key as `OPENAI_API_KEY`. For example:
     ```bash
     export OPENAI_API_KEY="sk-your-real-key"
     ```

Never commit real API keys to version control.

## Running the App

### Local execution

```bash
streamlit run app.py
```

Streamlit displays a local URL (typically `http://localhost:8501`). Open the link in your browser to interact with ISA.

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new Streamlit Cloud app pointing to `app.py`.
3. Open **App settings → Secrets** and paste the contents of your `.streamlit/secrets.toml` file.
4. Deploy. Streamlit Cloud automatically installs `requirements.txt` and launches the app.

## Using the Interface

### Main panel

- **User question** – Enter the prompt you want to send to the selected models. The text area pre-populates with a default prompt if one is defined in secrets.
- **Generate responses** – Sends the prompt to each selected model. Errors (missing prompt, missing model, API failures) are shown inline.
- **Response columns** – Each chosen model renders a Markdown response. The layout adapts to the number of models (up to two).

### Sidebar controls

- **Models to query** – Multiselect capped at two entries. Defaults to the first two items in the internal `DEFAULT_MODELS` list.
- **Temperature** – Controls randomness (`0.0` deterministic, `2.0` very creative).
- **Top P** – Nucleus sampling cutoff for additional randomness control.
- **Reasoning effort** – Maps to `reasoning.effort` for models that support structured reasoning (`gpt-4.1*`, `gpt-5*`).
- **Response verbosity** – Sets `text.verbosity` for eligible models (`low`, `medium`, or `high`).

### Viewing raw responses

Each response column includes a **“Show raw response”** expander containing the exact JSON returned by the Responses API. Use this to audit metadata such as usage, reasoning traces, or tool invocations.

## Customization

### Default prompts

The app looks for optional entries under the `[prompts]` section of `secrets.toml`:

```toml
[prompts]
developer_prompt = "Act as a friendly mentor..."
default_user_prompt = "How can I improve my study habits?"
```

- `developer_prompt` populates the developer message that precedes each user question.
- `default_user_prompt` pre-fills the user text area when the app loads.

### Supported models

`DEFAULT_MODELS` defines the options shown in the sidebar:

```python
DEFAULT_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]
```

All selected models must support the OpenAI Responses API. Models beginning with `gpt-4o` currently skip the reasoning and verbosity options; other models receive the chosen `reasoning.effort` and `text.verbosity` values.

To experiment with additional models, modify the list in `app.py` or re-enable the commented custom model input in `render_sidebar()`.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| **“OpenAI API key not found” error** | Ensure your key is set in `.streamlit/secrets.toml` or exported as `OPENAI_API_KEY` before launching the app. |
| **Blank page after clicking “Generate responses”** | Confirm a prompt is entered and at least one model is selected. The app stops execution when validations fail. |
| **`APIConnectionError` or `APIError` displayed** | Retry the request. If the error persists, review your network connection, model availability, and OpenAI account limits. |
| **Non-English validation messages** | Some warnings (e.g., missing prompt or models) appear in German per product requirements. |

## Development Notes

- The page metadata is set via `st.set_page_config(page_title="Intelligent Search Assistant", layout="wide")`.
- Responses are rendered as Markdown; consider sanitizing or adapting the rendering if model outputs include HTML.
- All API calls are made with `store=True` so that responses persist in the OpenAI console history.
- If you reintroduce the custom model input in `render_sidebar()`, maintain the two-model limit to preserve layout clarity.

