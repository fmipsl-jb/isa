"""*Staging* Intelligent Search Assistant"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import streamlit as st
from openai import APIConnectionError, APIError, OpenAI


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


@dataclass
class RunConfig:
    model: str
    prompt: str
    developer: Optional[str]
    temperature: float
    top_p: Optional[float]
    reasoning_effort: Optional[str]
    verbosity: str
    token: Optional[str]
    language: Optional[str]
    agent_type: str
    conversation_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    prompt_reference: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None


@dataclass
class ClassifierResult:
    token: Optional[str]
    language: Optional[str]


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


def build_input_messages(
    prompt: str,
    developer: Optional[str],
    *,
    include_developer: bool = True,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if developer and include_developer:
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


def extract_prompt_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, Sequence):
        chunks: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                chunks.append(text_value.strip())
        if chunks:
            return "\n".join(chunks)

    return ""


def extract_prompt_text(prompt_data: Dict[str, Any]) -> str:
    content_candidates: List[Any] = []

    if "content" in prompt_data:
        content_candidates.append(prompt_data.get("content"))

    prompt_section = prompt_data.get("prompt")
    if isinstance(prompt_section, dict):
        content_candidates.append(prompt_section.get("content"))

    data_section = prompt_data.get("data")
    if isinstance(data_section, dict):
        if "content" in data_section:
            content_candidates.append(data_section.get("content"))
        nested_prompt = data_section.get("prompt")
        if isinstance(nested_prompt, dict):
            content_candidates.append(nested_prompt.get("content"))

    for candidate in content_candidates:
        extracted = extract_prompt_text_from_content(candidate)
        if extracted:
            return extracted

    return ""


def load_default_user_prompt(client: OpenAI) -> str:
    prompts_section = st.secrets.get("prompts", {})
    prompt_id_raw = prompts_section.get("default_user_prompt_id")
    prompt_id = prompt_id_raw.strip() if isinstance(prompt_id_raw, str) else None
    if not prompt_id:
        return ""

    try:
        prompt_data: Dict[str, Any] = client.get(
            f"/prompts/{prompt_id}",
            cast_to=dict,
        )
    except Exception as error:  # pylint: disable=broad-except
        st.warning(f"Failed to load default prompt: {error}")
        return ""

    prompt_text = extract_prompt_text(prompt_data)
    if prompt_text:
        return prompt_text

    st.warning("Default prompt did not contain any text content.")
    return ""


def get_prompt_config(name: str) -> Dict[str, Any]:
    prompts_section = st.secrets.get("prompts", {})
    config: Dict[str, Any] = {}

    nested = prompts_section.get(name)
    if isinstance(nested, Mapping):
        config.update(nested)

    prefix = f"{name}_"
    for key, value in prompts_section.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        config[key[len(prefix) :]] = value

    return config


def parse_variable_names(value: Any) -> List[str]:
    if isinstance(value, str):
        return [entry.strip() for entry in value.split(",") if entry.strip()]

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        names: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
        return names

    return []


def build_prompt_reference(
    name: str,
    user_prompt: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    prompt_config = dict(config or get_prompt_config(name))
    prompt_id_raw = prompt_config.get("id")
    prompt_id = str(prompt_id_raw).strip() if prompt_id_raw else ""
    if not prompt_id:
        return None

    prompt_reference: Dict[str, Any] = {"id": prompt_id}

    variable_candidates = (
        prompt_config.get("variable_names")
        or prompt_config.get("variable_name")
        or prompt_config.get("input_variable")
    )
    variable_names = parse_variable_names(variable_candidates)

    prompt_variables: Dict[str, Any] = {}
    if variable_names:
        for variable_name in variable_names:
            prompt_variables[variable_name] = user_prompt
    else:
        for fallback in ("user_input", "input", "query", "question"):
            prompt_variables.setdefault(fallback, user_prompt)

    extra_variables = prompt_config.get("variables")
    if isinstance(extra_variables, Mapping):
        for key, value in extra_variables.items():
            if isinstance(key, str) and key:
                prompt_variables.setdefault(key, value)

    if prompt_variables:
        prompt_reference["variables"] = prompt_variables

    return prompt_reference


def model_supports_reasoning_and_verbosity(model: str) -> bool:
    """Return True when the model accepts reasoning and verbosity parameters."""
    normalized = model.lower().strip()
    return normalized.startswith("gpt-5")


def model_supports_top_p(model: str) -> bool:
    """Return True when the model accepts the `top_p` parameter."""
    normalized = model.lower().strip()
    return not normalized.startswith("gpt-5")


def model_supports_streaming(model: str) -> bool:
    """Return True when the model can be queried using streaming responses."""
    normalized = model.lower().strip()
    return not normalized.startswith("gpt-5")


def run_model(
    client: OpenAI,
    config: RunConfig,
    *,
    stream: bool = False,
    on_text_delta: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], str]:
    include_developer = not (config.conversation_id or config.previous_response_id)
    input_messages = build_input_messages(
        config.prompt,
        config.developer,
        include_developer=include_developer,
    )
    supports_reasoning_and_verbosity = model_supports_reasoning_and_verbosity(config.model)
    params: Dict[str, Any] = {
        "model": config.model,
        "input": input_messages,
        "text": prepare_text_config(
            config.verbosity if supports_reasoning_and_verbosity else None
        ),
        "store": True,
    }

    if config.cache_key:
        params["prompt_cache_key"] = config.cache_key

    metadata: Dict[str, Any] = {}
    if config.token:
        metadata["token"] = config.token
    if config.language:
        metadata["language"] = config.language
    if metadata:
        params["metadata"] = metadata

    if config.conversation_id:
        params["conversation"] = config.conversation_id
    elif config.previous_response_id:
        params["previous_response_id"] = config.previous_response_id

    if config.prompt_reference:
        params["prompt"] = config.prompt_reference

    tools: List[Dict[str, Any]] = []
    if config.agent_type in {"app", "creative"}:
        tools.append(
            {
                "type": "file_search",
                "vector_store_ids": ["vs_68c92dcc842c81919b9996ec34b55c2c"],
            }
        )

    if config.agent_type == "creative":
        tools.append(
            {
                "type": "web_search",
                "filters": {
                    "allowed_domains": [
                        "support.presonus.com",
                        "www.presonus.com",
                    ]
                },
            }
        )
        params["include"] = ["web_search_call.action.sources"]

    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    if config.top_p is not None and model_supports_top_p(config.model):
        params["top_p"] = config.top_p

    if supports_reasoning_and_verbosity:
        if config.reasoning_effort and config.reasoning_effort != "default":
            params["reasoning"] = {"effort": config.reasoning_effort}
    else:
        params["temperature"] = config.temperature

    supports_streaming = model_supports_streaming(config.model)

    if stream and supports_streaming:
        text_chunks: List[str] = []

        def append_text(delta_value: Optional[str]) -> None:
            if not delta_value:
                return
            text_chunks.append(delta_value)
            if on_text_delta:
                on_text_delta("".join(text_chunks))

        def extract_delta_text(delta: Any) -> Optional[str]:
            if isinstance(delta, str):
                return delta

            if isinstance(delta, dict):
                text_value = delta.get("text")
                if isinstance(text_value, str):
                    return text_value
                return None

            text_attr = getattr(delta, "text", None)
            if isinstance(text_attr, str):
                return text_attr

            to_dict = getattr(delta, "to_dict", None)
            if callable(to_dict):
                delta_dict = to_dict()
                if isinstance(delta_dict, dict):
                    text_value = delta_dict.get("text")
                    if isinstance(text_value, str):
                        return text_value

            return None

        with client.responses.stream(**params) as stream_response:
            final_response = None
            for event in stream_response:
                if event.type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    append_text(extract_delta_text(delta))
                elif event.type == "response.error":
                    error = getattr(event, "error", None)
                    message = "Unexpected streaming error"
                    if isinstance(error, dict):
                        message = error.get("message", message)
                    elif isinstance(error, str):
                        message = error
                    raise RuntimeError(message)

            final_response = stream_response.get_final_response()

        response_dict = final_response.to_dict() if final_response else {}
        if not text_chunks:
            text_output = extract_output_text(response_dict)
        else:
            text_output = "".join(text_chunks)
        return response_dict, text_output

    response = client.responses.create(**params)
    response_dict = response.to_dict()
    text_output = extract_output_text(response_dict)
    return response_dict, text_output


def strip_code_fences(text: str) -> str:
    sanitized = text.strip()
    if not sanitized.startswith("```"):
        return sanitized

    lines = sanitized.splitlines()
    if len(lines) < 3:
        return sanitized

    if lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
        core = [line for line in lines[1:-1] if not line.strip().startswith("```")]
        return "\n".join(core).strip()

    return sanitized


def parse_classifier_response(output_text: str) -> Optional[ClassifierResult]:
    if not output_text:
        return None

    sanitized = strip_code_fences(output_text)
    try:
        payload = json.loads(sanitized)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, Mapping):
        return None

    token_raw = payload.get("token")
    token: Optional[str] = None
    if isinstance(token_raw, str) and token_raw.strip():
        token = token_raw.strip().upper()

    language_raw = payload.get("language")
    language: Optional[str] = None
    if isinstance(language_raw, str) and language_raw.strip():
        language = language_raw.strip().lower()

    if token or language:
        return ClassifierResult(token=token, language=language)

    return None


def classify_user_prompt(client: OpenAI, user_prompt: str) -> Optional[ClassifierResult]:
    prompt_config = get_prompt_config("prompt_classifier")
    prompt_reference = build_prompt_reference(
        "prompt_classifier", user_prompt, config=prompt_config
    )
    if not prompt_reference:
        return None

    model_raw = prompt_config.get("model")
    model = str(model_raw).strip() if model_raw else "gpt-4.1-nano"
    if not model:
        model = "gpt-4.1-nano"

    cache_key_raw = prompt_config.get("cache_key")
    cache_key = str(cache_key_raw).strip() if cache_key_raw else "isa-classifier"
    if not cache_key:
        cache_key = "isa-classifier"

    params: Dict[str, Any] = {
        "model": model,
        "input": build_input_messages(user_prompt, None, include_developer=False),
        "prompt": prompt_reference,
        "text": prepare_text_config(None),
        "store": True,
        "prompt_cache_key": cache_key,
        "temperature": 0.0,
    }

    try:
        response = client.responses.create(**params)
    except Exception:  # pylint: disable=broad-except
        return None

    response_dict = response.to_dict()
    output_text = extract_output_text(response_dict)
    return parse_classifier_response(output_text)


def determine_route(token: Optional[str]) -> str:
    if not token:
        return "route_1"

    normalized = token.strip().upper()
    if normalized in {"APP", "OOS"}:
        return "route_1"
    if normalized in {"CREATIVE", "HYBRID"}:
        return "route_2"

    return "route_1"


def resolve_metadata_token(route: str, token: Optional[str]) -> str:
    allowed = {"APP", "CREATIVE", "HYBRID", "OOS"}
    if token and token.strip().upper() in allowed:
        normalized = token.strip().upper()
        if route == "route_2" and normalized == "APP":
            return "CREATIVE"
        return normalized

    return "CREATIVE" if route == "route_2" else "APP"


def build_route_run_config(
    route: str,
    user_prompt: str,
    developer_prompt: Optional[str],
    *,
    conversation_id: Optional[str],
    previous_response_id: Optional[str],
    prompt_reference: Optional[Dict[str, Any]],
    cache_key_base: str,
    token: Optional[str],
    language: Optional[str],
) -> Tuple[RunConfig, bool]:
    base = cache_key_base.strip() if cache_key_base else "isa-app"
    if not base:
        base = "isa-app"

    if route == "route_2":
        return (
            RunConfig(
                model="gpt-5-nano",
                prompt=user_prompt,
                developer=developer_prompt,
                temperature=0.5,
                top_p=None,
                reasoning_effort="low",
                verbosity="low",
                token=token,
                language=language,
                agent_type="creative",
                conversation_id=conversation_id,
                previous_response_id=previous_response_id,
                prompt_reference=prompt_reference,
                cache_key=f"{base}-route-2",
            ),
            False,
        )

    return (
        RunConfig(
            model="gpt-4.1-nano",
            prompt=user_prompt,
            developer=developer_prompt,
            temperature=0.5,
            top_p=0.8,
            reasoning_effort="low",
            verbosity="low",
            token=token,
            language=language,
            agent_type="app",
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            prompt_reference=prompt_reference,
            cache_key=f"{base}-route-1",
        ),
        True,
    )


def render_conversation_history(model: str) -> bool:
    history: List[Dict[str, str]] = st.session_state["conversation_history"].get(
        model, []
    )
    if not history:
        return False

    for message in history:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        if not content:
            continue
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    return True


def extract_conversation_id(response: Dict[str, Any]) -> Optional[str]:
    conversation = response.get("conversation")
    if isinstance(conversation, dict):
        conversation_id = conversation.get("id")
        if isinstance(conversation_id, str) and conversation_id.strip():
            return conversation_id.strip()

    conversation_id = response.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id.strip():
        return conversation_id.strip()

    return None


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


def main() -> None:
    st.set_page_config(page_title="*Staging* Intelligent Search Assistant", layout="wide")
    st.title("*Staging* Intelligent Search Assistant")
    st.caption("version 3.0.3 (250929)")

    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = {}
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = None

    try:
        client = build_client()
    except RuntimeError:
        st.stop()

    # Configuration sidebar intentionally disabled for Version 3.
    # config = render_sidebar()

    default_user_prompt = load_default_user_prompt(client)
    prompt = st.text_area(
        "User question",
        value=default_user_prompt,
        height=120,
        placeholder="Ask a question in any language, such as \"¿Qué es la consola y cómo se utiliza?\" or \"Hoe bewerk ik meerkanaals drumopnames om de timing te optimaliseren?\"",
    )

    developer_prompt = load_developer_prompt() or None

    run_button = st.button("Generate responses", type="primary")

    user_prompt = prompt.strip()
    response_container = st.container()

    if run_button:
        if not user_prompt:
            st.warning("Please enter a question before generating a response.")
            st.stop()

        conversations: Dict[str, Any] = st.session_state["conversations"]
        active_model = st.session_state.get("active_model")
        conversation_state: Dict[str, Any] = {}
        metadata_token: Optional[str] = None
        metadata_language: Optional[str] = None
        route: str
        target_model: str

        if isinstance(active_model, str):
            state_entry = conversations.get(active_model)
            if (
                isinstance(state_entry, dict)
                and (
                    state_entry.get("conversation_id")
                    or state_entry.get("previous_response_id")
                )
            ):
                conversation_state = state_entry
                target_model = active_model
                route = state_entry.get("route") or (
                    "route_2" if active_model != "gpt-4.1-nano" else "route_1"
                )
                metadata_token = state_entry.get("token")
                metadata_language = state_entry.get("language")
            else:
                target_model = ""
                route = "route_1"
        else:
            target_model = ""
            route = "route_1"

        continuing_conversation = bool(conversation_state)

        if not continuing_conversation:
            classifier_result = classify_user_prompt(client, user_prompt)
            classifier_token = classifier_result.token if classifier_result else None
            route = determine_route(classifier_token)
            metadata_token = resolve_metadata_token(route, classifier_token)
            metadata_language = (
                classifier_result.language if classifier_result else None
            )
            target_model = "gpt-4.1-nano" if route == "route_1" else "gpt-5-nano"
            conversation_state = {}

        metadata_token = metadata_token or resolve_metadata_token(route, None)

        prompt_config_name = "prompt_app" if route == "route_1" else "prompt_creative"
        prompt_config = get_prompt_config(prompt_config_name)
        prompt_reference = build_prompt_reference(
            prompt_config_name, user_prompt, config=prompt_config
        )
        cache_key_raw = prompt_config.get("cache_key")
        default_cache_key = "isa-app" if route == "route_1" else "isa-creative"
        cache_key_base = (
            str(cache_key_raw).strip() if cache_key_raw else default_cache_key
        ) or default_cache_key

        conversation_id: Optional[str] = None
        previous_response_id: Optional[str] = None
        if conversation_state:
            conversation_id = conversation_state.get("conversation_id")
            previous_response_id = conversation_state.get("previous_response_id")

        run_config, stream_enabled = build_route_run_config(
            route,
            user_prompt,
            developer_prompt,
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            prompt_reference=prompt_reference,
            cache_key_base=cache_key_base,
            token=metadata_token,
            language=metadata_language,
        )

        st.session_state["active_model"] = target_model

        with response_container:
            st.subheader(f"Response ({target_model})")
            render_conversation_history(target_model)

            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                output_placeholder = st.empty()

        def update_streaming_text(text: str) -> None:
            output_placeholder.markdown(text if text else " ")

        response: Dict[str, Any]
        output_text = ""

        with response_container:
            with st.spinner("Retrieving response…"):
                try:
                    response, output_text = run_model(
                        client,
                        run_config,
                        stream=stream_enabled,
                        on_text_delta=update_streaming_text if stream_enabled else None,
                    )
                except (APIConnectionError, APIError) as api_error:
                    output_placeholder.error(f"API error: {api_error}")
                    return
                except Exception as error:  # pylint: disable=broad-except
                    output_placeholder.error(f"Unexpected error: {error}")
                    return

        sanitized_output = output_text.strip()
        if sanitized_output:
            output_placeholder.markdown(output_text)
        else:
            output_placeholder.markdown("_No output text was returned._")

        conversation_identifier = extract_conversation_id(response)
        existing_state: Dict[str, Any] = {}
        if continuing_conversation and isinstance(conversation_state, dict):
            existing_state = conversation_state

        new_state: Dict[str, Any] = {
            "conversation_id": conversation_identifier
            or existing_state.get("conversation_id"),
            "previous_response_id": response.get("id"),
            "route": route,
            "token": metadata_token or existing_state.get("token"),
            "language": metadata_language or existing_state.get("language"),
        }
        st.session_state["conversations"][target_model] = new_state

        model_history = st.session_state["conversation_history"].setdefault(
            target_model, []
        )
        model_history.extend(
            [
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": sanitized_output
                    if sanitized_output
                    else "_No output text was returned._",
                },
            ]
        )

        with response_container:
            with st.expander("Show raw response"):
                st.json(response)
    else:
        active_model = st.session_state.get("active_model")
        with response_container:
            if active_model:
                st.subheader(f"Response ({active_model})")
                has_history = render_conversation_history(active_model)
                if not has_history:
                    st.info("No conversation yet. Ask a question to get started.")
            else:
                st.info("No conversation yet. Ask a question to get started.")


if __name__ == "__main__":
    main()
