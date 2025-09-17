# ISA Streamlit App

Streamlit playground for testing prompts and OpenAI Responses API models side by side.

## To-Dos

1. **Create a virtual environment (optional but recommended).**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install the dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
3. **Add your OpenAI API key.**
   - Duplicate `secrets.example.toml` (or `.streamlit/secrets.template.toml`) to `.streamlit/secrets.toml`.
   - Replace the placeholder value with your real key.
   - Alternatively, set the `OPENAI_API_KEY` environment variable—this is how Streamlit Community Cloud stores secrets.
4. **Launch the Streamlit app.**
   ```bash
   streamlit run app.py
   ```
5. **Open the provided local URL in your browser.**
6. **Enter your question in the “User question” box.**
7. **Adjust the developer prompt or leave the default if you want to use the Studio One mentor persona.**
8. **Pick up to two models in the sidebar (add a custom one if needed).**
9. **Tune temperature, max tokens, reasoning effort, and verbosity in the sidebar.**
10. **Press “Generate responses” to send the question to the selected models.**
11. **Compare the answers side by side and open the “Show raw response” expander for the API JSON if required.**

## Notes

- Secrets must always live in `.streamlit/secrets.toml`; never commit real API keys.
- The app expects models compatible with the Responses API (e.g., `gpt-4.1-mini`, `gpt-4o-mini`).
- Use the sidebar to remove or swap models whenever you need to test a different pair.
