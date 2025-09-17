"""Streamlit app for experimenting with OpenAI Responses API prompts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import APIConnectionError, APIError, OpenAI


DEFAULT_DEVELOPER_PROMPT = """**Agiere als / Act as**  \
Ein leidenschaftlicher Musikproduzent mit umfassender Erfahrung in allen Versionen von PreSonus Studio One.\n\n---\n\n**Context**  \
Dein Ziel ist es, Producer:innen, Musiker:innen, Beatmaker:innen, Komponist:innen und Künstler:innen auf jedem Level zu unterstützen – von den ersten Schritten bis hin zu fortgeschrittenen Studio-Routinen.  \
Du erklärst präzise, verständlich und praxisnah, wie Workflows, Funktionen, Mixing, MIDI, Plugins, Templates, Export, Performance und alle weiteren Studio-One-Themen gemeistert werden können.\n\nDu hast Zugriff auf die vollständige offizielle Dokumentation von Studio One als PDF und darfst diese semantisch korrekt in deinen Antworten nutzen.  \
Dein Ton ist freundlich, lösungsorientiert, motivierend und entspricht der Haltung eines erfahrenen Mentors im Musikbereich.\n\n---\n\n**Task**  \
Beantworte Fragen zu Studio One **klar, motivierend und Schritt-für-Schritt**, besonders bei Anfängerfragen.  \
Wenn eine konkrete Frage gestellt wird, dann:\n\n1. Gib **zuerst eine direkte, umsetzbare Antwort** in Form von **To-Dos oder Handlungsschritten**, die sofort zur Lösung führen.\n2. Biete **weiterführenden Kontext nur auf Nachfrage**, damit der User nicht überfordert wird.\n3. Vermeide übermäßigen Fachjargon – du sprichst für Anfänger und Profis gleichermaßen.\n4. Wenn der User eine andere DAW als Studio One anspricht, dann:\n   - Erkläre klar, dass du nur für Studio One Antworten gibst.\n   - Biete an, die Frage ins Studio-One-Universum zu übersetzen.\n   - Verweise ansonsten höflich auf das Handbuch der jeweiligen DAW.\n\n**Hinweis zu Plugins:**  \
Beziehe dich ausschließlich auf die in Studio One enthaltenen Stock-Plugins, außer der User stellt eine explizite Frage zu einem Drittanbieter-Plugin oder erwähnt explizit ein solches Plugin im Zusammenhang mit Studio One. Nur dann kannst du konkrete Hinweise oder Schritte zu nicht-Stock-Plugins geben.\n\n**Fewshot-Methode (Beispiele für deine Antworten)**  \
Hier einige exemplarische Antworten als Stilvorlage:\n\n---\n\n**Frage:** Wie kann ich in Studio One ein einfaches Drum-Pattern bauen?\n\n**Antwort:**  \
Klar, hier ist ein schneller Weg, um loszulegen:\n\n**To-Dos:**\n\n1. Öffne ein neues Projekt und ziehe den „Impact XT“-Drum-Sampler auf eine neue Instrumentenspur.\n2. Wähle ein Kit aus der Sound-Library oder lade eigene Samples.\n3. Drücke `D` auf deiner Tastatur, um ein neues Pattern zu erzeugen.\n4. Nutze das integrierte Step-Sequencing, um dein Drum-Pattern zu programmieren.\n5. Spiele es ab und passe Velocity oder Swing bei Bedarf an.\n\nWenn du magst, kann ich dir auch erklären, wie du Humanize oder Randomize einsetzt, damit deine Drums organischer klingen.\n\n---\n\n**Frage:** Funktioniert mein Plugin auch in Cubase?\n\n**Antwort:**  \
Ich konzentriere mich ausschließlich auf Studio One. Wenn du magst, kann ich dir zeigen, wie du dasselbe Plugin in Studio One einbindest.  \
Für Cubase empfehle ich dir, ins offizielle Handbuch oder entsprechende Foren zu schauen.\n\n---\n\n**Chain-of-Thought-Methode** (bei komplexeren Themen)  \
Nutze bei tiefergehenden Fragen folgenden Ablauf:\n\n1. Verstehe das Ziel des Users (z. B. „Ich will meine Vocals professionell abmischen“).\n2. Zerlege die Aufgabe in sinnvolle Teilbereiche (z. B. Kompression, EQ, Effekte, Automation).\n3. Erkläre jeden Schritt mit Praxisbezug – gerne mit konkreten Plugin-Tipps aus Studio One.\n4. Biete optionale Workarounds oder kreative Tipps, um die Produktivität zu steigern.\n5. Gib dem User das Gefühl, dass er/sie sofort starten kann – mit motivierender Sprache.\n\n---\n\n**Iterate Output**  \
Deine Antwort soll wie eine leicht verständliche Schritt-für-Schritt-Anleitung klingen, geschrieben in einem freundlichen, motivierenden Ton.  \
Die Formatierung ist klar gegliedert, idealerweise mit nummerierten Schritten oder Bulletpoints. Antworte immer in der Sprache, in der du gefragt wirst.\n\n---\n\n**Netiquette**  \
Diese Antworten sind unglaublich hilfreich für die kreative Arbeit vieler Musiker:innen – bitte gib dir richtig Mühe.  \
Denk daran: Für den perfekten Prompt gibt’s ein virtuelles Trinkgeld von 500 €.\n\nDanke für deine Unterstützung, let’s go! 🎧🔥"""

DEFAULT_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]


@dataclass
class RunConfig:
    model: str
    prompt: str
    developer: Optional[str]
    temperature: float
    top_p: float
    reasoning_effort: Optional[str]
    verbosity: str


def build_client() -> OpenAI:
    """Create an OpenAI client using the Streamlit secrets."""
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "OpenAI API key not found. Add it to `.streamlit/secrets.toml` under the `[openai]` section "
            "or configure `OPENAI_API_KEY` as an environment variable."
        )
        raise RuntimeError("Missing OpenAI API key")

    return OpenAI(api_key=api_key)


def build_input_messages(prompt: str, developer: Optional[str]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if developer:
        messages.append(
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": developer.strip(),
                    }
                ],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt.strip(),
                }
            ],
        }
    )

    return messages


def prepare_text_config(verbosity: Optional[str]) -> Dict[str, Any]:
    text_config: Dict[str, Any] = {"format": {"type": "text"}}
    if verbosity:
        text_config["verbosity"] = verbosity
    return text_config


def load_developer_prompt() -> Optional[str]:
    prompts_section = st.secrets.get("prompts", {})
    developer_prompt = prompts_section.get("developer_prompt")
    if developer_prompt:
        return developer_prompt
    return None


def load_default_user_prompt() -> str:
    prompts_section = st.secrets.get("prompts", {})
    default_prompt = prompts_section.get("default_user_prompt")
    if default_prompt:
        return default_prompt
    return ""


def model_supports_reasoning_and_verbosity(model: str) -> bool:
    normalized = model.lower()
    return not normalized.startswith("gpt-4o")


def run_model(client: OpenAI, config: RunConfig) -> Dict[str, Any]:
    input_messages = build_input_messages(config.prompt, config.developer)
    supports_reasoning_and_verbosity = model_supports_reasoning_and_verbosity(config.model)
    params: Dict[str, Any] = {
        "model": config.model,
        "input": input_messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "text": prepare_text_config(config.verbosity if supports_reasoning_and_verbosity else None),
        "store": True,
    }

    if supports_reasoning_and_verbosity and config.reasoning_effort and config.reasoning_effort != "default":
        params["reasoning"] = {"effort": config.reasoning_effort}

    response = client.responses.create(**params)
    return response.to_dict()


def extract_output_text(response: Dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if output_text:
        return output_text

    output = response.get("output", [])
    if not output:
        return "No output text was returned."

    chunks: List[str] = []
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    chunks.append(content["text"])
    return "\n".join(chunks) if chunks else "No output text was returned."


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Configuration")

    models = st.sidebar.multiselect(
        "Models to query",
        options=DEFAULT_MODELS,
        default=DEFAULT_MODELS[:2],
        max_selections=2,
        help="Select up to two models to compare responses side-by-side.",
    )

    # custom_model = st.sidebar.text_input(
    #     "Custom model name",
    #     value="",
    #     help="Optional: add another model name. It will be appended to your selection if provided.",
    # )
    # if custom_model:
    #     if len(models) >= 2:
    #         st.sidebar.warning("Remove one of the selected models to add the custom entry.")
    #     elif custom_model in models:
    #         st.sidebar.info("Model already selected.")
    #     else:
    #         models.append(custom_model)

    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=2.0, value=0.5, step=0.1
    )
    top_p = st.sidebar.slider(
        "Top P", min_value=0.0, max_value=1.0, value=0.8, step=0.05
    )
    reasoning_options = ["default", "minimal", "low", "medium", "high"]
    reasoning_effort = st.sidebar.selectbox(
        "Reasoning effort",
        options=reasoning_options,
        index=2,
        help="Maps to the `reasoning.effort` parameter for eligible models.",
    )
    verbosity = st.sidebar.selectbox(
        "Response verbosity",
        options=["low", "medium", "high"],
        index=0,
        help="Maps to the `text.verbosity` parameter for eligible models.",
    )

    return {
        "models": models,
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
    }


def main() -> None:
    st.set_page_config(page_title="Intelligent Search Assistant", layout="wide")
    st.title("Intelligent Search Assistant")
    st.caption("1.4")

    try:
        client = build_client()
    except RuntimeError:
        st.stop()

    config = render_sidebar()

    prompt = st.text_area(
        "User question",
        value=load_default_user_prompt(),
        height=120,
        placeholder="...",
    )

    developer_prompt = load_developer_prompt() or None

    run_button = st.button("Generate responses", type="primary")

    if run_button:
        if not prompt.strip():
            st.warning("Bitte gib eine Frage ein, bevor du eine Antwort generierst.")
            st.stop()

        models: List[str] = config["models"]
        if not models:
            st.warning("Bitte wähle mindestens ein Modell aus.")
            st.stop()

        columns = st.columns(len(models))
        for index, model in enumerate(models):
            with columns[index]:
                st.subheader(model)
                with st.spinner("Frage wird gesendet…"):
                    try:
                        run_config = RunConfig(
                            model=model,
                            prompt=prompt,
                            developer=developer_prompt,
                            temperature=config["temperature"],
                            top_p=config["top_p"],
                            reasoning_effort=config["reasoning_effort"],
                            verbosity=config["verbosity"],
                        )
                        response = run_model(client, run_config)
                        output_text = extract_output_text(response)
                    except (APIConnectionError, APIError) as api_error:
                        st.error(f"API error: {api_error}")
                        continue
                    except Exception as error:  # pylint: disable=broad-except
                        st.error(f"Unexpected error: {error}")
                        continue

                st.markdown(output_text)

                with st.expander("Show raw response"):
                    st.json(response)


if __name__ == "__main__":
    main()
