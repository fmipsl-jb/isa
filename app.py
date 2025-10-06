"""Studio Pro Assistant"""

from __future__ import annotations

import base64
import json
from urllib.parse import urlparse
import base64
import binascii
import io
import time
import uuid
import wave
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import av
from av.audio.resampler import AudioResampler
import numpy as np
import streamlit as st
from openai import APIConnectionError, APIError, OpenAI
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback for older Streamlit
    add_script_run_ctx = None  # type: ignore[assignment]
    get_script_run_ctx = None  # type: ignore[assignment]


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


def coerce_boolean(value: Any) -> Optional[bool]:
    """Attempt to coerce a value into a boolean."""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

    return None


def is_developer_mode_enabled() -> bool:
    """Return True when developer controls should be displayed."""

    app_section = st.secrets.get("app", {})
    if isinstance(app_section, Mapping):
        raw_flag = app_section.get("developer_mode")
        coerced = coerce_boolean(raw_flag)
        if coerced is not None:
            return coerced

    env_flag = os.getenv("ISA_DEVELOPER_MODE")
    if env_flag is not None:
        coerced = coerce_boolean(env_flag)
        if coerced is not None:
            return coerced

    return False


def parse_ice_servers(value: Any) -> List[str]:
    """Normalize ICE server definitions into a list of strings."""

    if isinstance(value, str):
        normalized = value.replace("\r", "\n")
        entries = [
            candidate.strip()
            for chunk in normalized.split("\n")
            for candidate in chunk.split(",")
            if candidate.strip()
        ]
        return entries

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        entries: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                entries.append(item.strip())
        return entries

    return []


def get_realtime_config() -> Dict[str, Any]:
    """Retrieve realtime configuration defaults from secrets."""

    defaults: Dict[str, Any] = {
        "model": "gpt-4o-realtime-preview",
        "voice": "verse",
        "sample_rate_hz": 16000,
        "latency_mode": "default",
        "ice_servers": [],
    }

    realtime_section = st.secrets.get("realtime", {})
    if not isinstance(realtime_section, Mapping):
        return dict(defaults)

    config = dict(defaults)

    model = realtime_section.get("model")
    if isinstance(model, str) and model.strip():
        config["model"] = model.strip()

    voice = realtime_section.get("voice")
    if isinstance(voice, str) and voice.strip():
        config["voice"] = voice.strip()

    latency_mode = realtime_section.get("latency_mode")
    if isinstance(latency_mode, str) and latency_mode.strip():
        config["latency_mode"] = latency_mode.strip()

    sample_rate = realtime_section.get("sample_rate_hz")
    if isinstance(sample_rate, (int, float)):
        sample_rate_int = int(sample_rate)
        if sample_rate_int > 0:
            config["sample_rate_hz"] = sample_rate_int
    elif isinstance(sample_rate, str):
        try:
            sample_rate_int = int(sample_rate.strip())
        except ValueError:
            sample_rate_int = 0
        if sample_rate_int > 0:
            config["sample_rate_hz"] = sample_rate_int

    ice_servers = parse_ice_servers(realtime_section.get("ice_servers"))
    if ice_servers:
        config["ice_servers"] = ice_servers

    return config


def is_valid_url(value: str) -> bool:
    """Return True when the provided value resembles a URL."""

    if not value or not isinstance(value, str):
        return False

    parsed = urlparse(value)
    return bool(parsed.scheme and (parsed.netloc or parsed.path))


def validate_realtime_config(config: Mapping[str, Any]) -> Optional[str]:
    """Validate realtime configuration values before starting a session."""

    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        return "Realtime model must be provided."

    voice = config.get("voice")
    if not isinstance(voice, str) or not voice.strip():
        return "Voice must be provided."

    latency_mode = config.get("latency_mode")
    if not isinstance(latency_mode, str) or not latency_mode.strip():
        return "Latency mode must be provided."

    sample_rate = config.get("sample_rate_hz")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        return "Sample rate must be a positive integer."

    ice_servers = config.get("ice_servers", [])
    if not isinstance(ice_servers, Sequence) or isinstance(
        ice_servers, (str, bytes, bytearray)
    ):
        return "ICE servers must be provided as a list of URLs."

    for entry in ice_servers:
        if not isinstance(entry, str) or not entry.strip():
            return "ICE server entries cannot be empty."
        if not is_valid_url(entry.strip()):
            return f"ICE server '{entry}' is not a valid URL."

    return None


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
    daw: Optional[str]
    agent_type: str
    conversation_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    prompt_reference: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None
    mode: str = "text"
    voice_session_id: Optional[str] = None
    voice_role: str = "assistant"
    voice_session_id: Optional[str] = None
    mode: Optional[str] = None


@dataclass
class ClassifierResult:
    token: Optional[str]
    language: Optional[str]


@dataclass
class VoiceSessionMetadata:
    session_id: str
    client_secret: str
    ice_servers: Sequence[Mapping[str, Any]]
    model: str
    voice: str
    expires_at: Optional[str] = None


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


def _get_realtime_sessions_resource(client: OpenAI) -> Any:
    """Return the realtime sessions resource from the OpenAI client."""

    realtime = getattr(client, "realtime", None)
    sessions = getattr(realtime, "sessions", None)
    if sessions and hasattr(sessions, "create"):
        return sessions

    beta = getattr(client, "beta", None)
    realtime_beta = getattr(beta, "realtime", None)
    sessions = getattr(realtime_beta, "sessions", None) if realtime_beta else None
    if sessions and hasattr(sessions, "create"):
        return sessions

    raise AttributeError("OpenAI client does not expose realtime.sessions")


def _extract_client_secret(session_payload: Mapping[str, Any]) -> Optional[str]:
    candidate = session_payload.get("client_secret")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    if isinstance(candidate, Mapping):
        for key in ("value", "secret", "client_secret"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _ensure_mapping(payload: Any) -> Mapping[str, Any]:
    """Best-effort conversion of SDK objects into dictionaries."""

    if isinstance(payload, Mapping):
        return payload

    for attr in ("model_dump", "dict", "to_dict"):
        extractor = getattr(payload, attr, None)
        if callable(extractor):
            result = extractor()
            if isinstance(result, Mapping):
                return result

    return {}


def create_voice_session(
    client: OpenAI,
    *,
    model: str = "gpt-4o-realtime-preview",
    voice: str = "verse",
) -> VoiceSessionMetadata:
    sessions_resource = _get_realtime_sessions_resource(client)
    session_response = sessions_resource.create(
        model=model,
        voice=voice,
        modalities=["text", "audio"],
        input_audio_format="pcm16",
        output_audio_format="pcm16",
    )

    session_payload = _ensure_mapping(session_response)
    if not session_payload:
        raise RuntimeError("Realtime session request did not return metadata.")

    client_secret = _extract_client_secret(session_payload)
    if not client_secret:
        webrtc_payload = _ensure_mapping(session_payload.get("webrtc"))
        client_secret = _extract_client_secret(webrtc_payload)
    if not client_secret:
        raise RuntimeError("Realtime session did not include a client secret.")

    ice_servers_raw = session_payload.get("ice_servers")
    ice_servers: Sequence[Mapping[str, Any]]
    if isinstance(ice_servers_raw, Sequence):
        ice_servers = [entry for entry in ice_servers_raw if isinstance(entry, Mapping)]
    else:
        webrtc_payload = _ensure_mapping(session_payload.get("webrtc"))
        alt_ice_servers = webrtc_payload.get("ice_servers") or webrtc_payload.get("iceServers")
        if isinstance(alt_ice_servers, Sequence):
            ice_servers = [entry for entry in alt_ice_servers if isinstance(entry, Mapping)]
        else:
            ice_servers = []

    session_identifier = session_payload.get("id")
    if not isinstance(session_identifier, str) or not session_identifier.strip():
        raise RuntimeError("Realtime session did not provide an identifier.")

    return VoiceSessionMetadata(
        session_id=session_identifier.strip(),
        client_secret=client_secret,
        ice_servers=ice_servers,
        model=str(session_payload.get("model") or model),
        voice=str(session_payload.get("voice") or voice),
        expires_at=session_payload.get("expires_at"),
    )


def decode_audio_delta(delta: Any) -> Optional[av.AudioFrame]:
    if delta is None:
        return None

    if not isinstance(delta, Mapping):
        to_dict = getattr(delta, "to_dict", None)
        if callable(to_dict):
            delta_dict = to_dict()
            if isinstance(delta_dict, Mapping):
                delta = delta_dict
        else:
            return None

    audio_payload = delta.get("audio") or delta.get("pcm") or delta.get("data")
    if not isinstance(audio_payload, str) or not audio_payload:
        return None

    try:
        pcm_bytes = base64.b64decode(audio_payload)
    except (TypeError, ValueError):
        return None

    sample_rate_raw = delta.get("sample_rate") or delta.get("sampleRate") or 16000
    try:
        sample_rate = int(sample_rate_raw)
    except (TypeError, ValueError):
        sample_rate = 16000

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return None

    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = sample_rate
    return frame


def create_silent_frame(sample_count: int, sample_rate: int) -> av.AudioFrame:
    if sample_count <= 0:
        sample_count = int(sample_rate / 10)
    zeros = np.zeros((1, sample_count), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(zeros, format="s16", layout="mono")
    frame.sample_rate = sample_rate
    return frame


def _consume_remote_audio(
    stream_response: Any,
    playback_queue: "queue.Queue[av.AudioFrame]",
    error_queue: "queue.Queue[str]",
) -> None:
    queue = st.session_state.setdefault("realtime_event_queue", [])

    def _event_to_mapping(candidate: Any) -> Mapping[str, Any]:
        if isinstance(candidate, Mapping):
            return candidate
        for attr in ("model_dump", "dict", "to_dict"):
            extractor = getattr(candidate, attr, None)
            if callable(extractor):
                try:
                    mapping = extractor()
                except Exception:  # pylint: disable=broad-except
                    continue
                if isinstance(mapping, Mapping):
                    return mapping
        event_type = getattr(candidate, "type", None)
        return {"type": event_type} if event_type else {}

    try:
        for event in stream_response:
            event_type = getattr(event, "type", "")
            event_payload = _event_to_mapping(event)

            if event_type in {
                "response.output_text.delta",
                "response.output_audio.delta",
                "response.created",
                "response.completed",
            }:
                queue.append(event_payload)
                if event_type == "response.completed":
                    entry = process_realtime_completed_event(event_payload)
                    if entry is not None:
                        event_payload["_entry"] = entry
                else:
                    process_realtime_delta_event(event_payload)
                    event_payload["_processed_delta"] = True

            if event_type == "response.output_audio.delta":
                delta = getattr(event, "delta", None)
                if not delta and isinstance(event_payload, Mapping):
                    delta = event_payload.get("delta")
                frame = decode_audio_delta(delta)
                if frame is not None:
                    playback_queue.put(frame)
            elif event_type in {"response.output_audio.done", "response.completed"}:
                break
            elif event_type == "response.error":
                error_obj = getattr(event, "error", None)
                message: str
                if isinstance(error_obj, Mapping):
                    message_candidate = error_obj.get("message")
                    message = (
                        message_candidate
                        if isinstance(message_candidate, str)
                        else "Unexpected realtime error"
                    )
                elif error_obj:
                    message = str(error_obj)
                else:
                    message = "Unexpected realtime error"
                error_queue.put(message)
                break
    except Exception as error:  # pylint: disable=broad-except
        error_queue.put(str(error))


def ensure_voice_stream(
    client: OpenAI,
    metadata: VoiceSessionMetadata,
) -> Optional[Any]:
    stream_response = st.session_state.get("voice_stream_response")
    if stream_response is not None:
        return stream_response

    params: Dict[str, Any] = {
        "model": metadata.model,
        "session": {
            "id": metadata.session_id,
            "client_secret": metadata.client_secret,
        },
        "modalities": ["text", "audio"],
        "audio": {"voice": metadata.voice, "format": "pcm16"},
    }

    try:
        stream_context = client.responses.stream(**params)
    except Exception as error:  # pylint: disable=broad-except
        st.session_state["voice_session_error"] = str(error)
        return None

    stream_response = stream_context.__enter__()
    st.session_state["voice_stream_context"] = stream_context
    st.session_state["voice_stream_response"] = stream_response

    playback_queue = st.session_state.get("voice_remote_queue")
    if not isinstance(playback_queue, queue.Queue):
        playback_queue = queue.Queue()
        st.session_state["voice_remote_queue"] = playback_queue

    error_queue = st.session_state.get("voice_remote_error_queue")
    if not isinstance(error_queue, queue.Queue):
        error_queue = queue.Queue()
        st.session_state["voice_remote_error_queue"] = error_queue

    listener_thread = threading.Thread(
        target=_consume_remote_audio,
        args=(stream_response, playback_queue, error_queue),
        daemon=True,
    )
    if add_script_run_ctx is not None and get_script_run_ctx is not None:
        try:
            ctx = get_script_run_ctx()
        except RuntimeError:
            ctx = None
        if ctx is not None:
            try:
                add_script_run_ctx(listener_thread, ctx=ctx)
            except RuntimeError:
                pass
    listener_thread.start()
    st.session_state["voice_listener_thread"] = listener_thread

    return stream_response


def stop_voice_stream() -> None:
    stream_context = st.session_state.pop("voice_stream_context", None)
    if stream_context is not None:
        try:
            stream_context.__exit__(None, None, None)
        except Exception:  # pylint: disable=broad-except
            pass

    st.session_state.pop("voice_stream_response", None)
    st.session_state.pop("voice_remote_queue", None)
    st.session_state.pop("voice_remote_error_queue", None)
    st.session_state.pop("voice_listener_thread", None)
    st.session_state.pop("voice_session_client", None)
    st.session_state["voice_session_started"] = False
    st.session_state["voice_session_error"] = None


def stop_voice_session() -> None:
    stop_voice_stream()
    st.session_state["voice_session_active"] = False
    st.session_state["voice_session_metadata"] = None
    st.session_state["voice_session_error"] = None
    if "voice_session_toggle" in st.session_state:
        st.session_state["voice_session_toggle"] = False


class VoiceAudioProcessor(AudioProcessorBase):
    """Audio bridge between Streamlit's WebRTC stack and the Realtime API."""

    def __init__(self) -> None:
        self._metadata: Optional[VoiceSessionMetadata] = st.session_state.get(
            "voice_session_metadata"
        )
        self._client: Optional[OpenAI] = None
        self._stream_response: Optional[Any] = None
        playback_queue = st.session_state.get("voice_remote_queue")
        if isinstance(playback_queue, queue.Queue):
            self._playback_queue: "queue.Queue[av.AudioFrame]" = playback_queue
        else:
            self._playback_queue = queue.Queue()
            st.session_state["voice_remote_queue"] = self._playback_queue
        error_queue = st.session_state.get("voice_remote_error_queue")
        if isinstance(error_queue, queue.Queue):
            self._error_queue: "queue.Queue[str]" = error_queue
        else:
            self._error_queue = queue.Queue()
            st.session_state["voice_remote_error_queue"] = self._error_queue
        self._sample_rate = 16000
        self._audio_resampler: Optional[AudioResampler] = None

        if self._metadata is not None:
            self._client = st.session_state.get("voice_session_client")
            if self._client is None:
                try:
                    self._client = build_client()
                except RuntimeError:
                    self._client = None

            if self._client is not None:
                self._stream_response = ensure_voice_stream(
                    self._client, self._metadata
                )

    def _frame_to_pcm(self, frame: av.AudioFrame) -> bytes:
        samples = frame.to_ndarray()
        if samples.ndim > 1:
            samples = samples.mean(axis=0)
        pcm = np.asarray(samples, dtype=np.int16)
        return pcm.tobytes()

    def _prepare_playback_frame(
        self, frame: av.AudioFrame, sample_count: int, sample_rate: int
    ) -> av.AudioFrame:
        if sample_count <= 0:
            sample_count = max(int(sample_rate / 100), 1)
        frame_sample_rate = getattr(frame, "sample_rate", sample_rate) or sample_rate

        if frame_sample_rate != sample_rate:
            if (
                self._audio_resampler is None
                or getattr(self._audio_resampler, "rate", None) != sample_rate
            ):
                self._audio_resampler = AudioResampler(
                    format="s16", layout="mono", rate=sample_rate
                )
            try:
                resampled_frames = self._audio_resampler.resample(frame)
            except Exception:  # pylint: disable=broad-except
                resampled_frames = []
            if resampled_frames:
                frame = resampled_frames[0]
        samples = frame.to_ndarray()
        if samples.ndim > 1:
            samples = samples.mean(axis=0)
        pcm = np.asarray(samples, dtype=np.int16).reshape(-1)

        current_samples = pcm.shape[0]
        if current_samples > sample_count:
            pcm = pcm[:sample_count]
        elif current_samples < sample_count:
            pcm = np.pad(pcm, (0, sample_count - current_samples), mode="constant")

        output_frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        output_frame.sample_rate = sample_rate
        return output_frame

    def _drain_error_queue(self) -> Optional[str]:
        message: Optional[str] = None
        try:
            while True:
                message = self._error_queue.get_nowait()
        except queue.Empty:
            pass
        return message

    def _handle_error_state(self, sample_count: int) -> Optional[av.AudioFrame]:
        message = self._drain_error_queue()
        if message:
            st.session_state["voice_session_error"] = message
            stop_voice_stream()
            self._stream_response = None
            return create_silent_frame(sample_count, self._sample_rate)
        if st.session_state.get("voice_session_error"):
            if self._stream_response is not None:
                stop_voice_stream()
                self._stream_response = None
            return create_silent_frame(sample_count, self._sample_rate)
        return None

    def _ensure_client(self) -> Optional[OpenAI]:
        if self._client is not None:
            return self._client
        candidate = st.session_state.get("voice_session_client")
        if isinstance(candidate, OpenAI):
            self._client = candidate
            return self._client
        try:
            self._client = build_client()
        except RuntimeError:
            self._client = None
        return self._client

    def _ensure_stream(self) -> Optional[Any]:
        if self._stream_response is not None:
            return self._stream_response
        if self._metadata is None:
            return None
        client = self._ensure_client()
        if client is None:
            return None
        self._stream_response = ensure_voice_stream(client, self._metadata)
        return self._stream_response

    def _process_audio_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        sample_count = getattr(frame, "samples", None)
        if sample_count is None:
            sample_count = frame.to_ndarray().shape[-1]
        try:
            sample_count = int(sample_count)
        except (TypeError, ValueError):
            sample_count = frame.to_ndarray().shape[-1]
        self._sample_rate = getattr(frame, "sample_rate", None) or self._sample_rate

        error_frame = self._handle_error_state(sample_count)
        if error_frame is not None:
            return error_frame

        if self._metadata is None:
            return create_silent_frame(sample_count, self._sample_rate)

        stream_response = self._ensure_stream()
        if stream_response is None:
            error_frame = self._handle_error_state(sample_count)
            if error_frame is not None:
                return error_frame
            return create_silent_frame(sample_count, self._sample_rate)

        audio_buffer = getattr(stream_response, "input_audio_buffer", None)
        try:
            if audio_buffer is not None:
                audio_buffer.append(self._frame_to_pcm(frame))
                commit = getattr(audio_buffer, "commit", None)
                if callable(commit):
                    commit()
            response_controller = getattr(stream_response, "response", None)
            create_fn = getattr(response_controller, "create", None)
            if callable(create_fn):
                create_fn()
        except Exception as error:  # pylint: disable=broad-except
            st.session_state["voice_session_error"] = str(error)
            stop_voice_stream()
            self._stream_response = None
            return create_silent_frame(sample_count, self._sample_rate)

        try:
            playback_frame = self._playback_queue.get_nowait()
            return self._prepare_playback_frame(
                playback_frame, sample_count, self._sample_rate
            )
        except queue.Empty:
            return create_silent_frame(sample_count, self._sample_rate)

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:  # type: ignore[override]
        return self._process_audio_frame(frame)

    def recv_queued(  # type: ignore[override]
        self, frames: "List[av.AudioFrame]"
    ) -> av.AudioFrame:
        """Process queued frames when async audio processing is enabled."""
        if not frames:
            return create_silent_frame(int(self._sample_rate / 10), self._sample_rate)
        return self._process_audio_frame(frames[-1])


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
    if config.mode == "voice":
        return run_voice_session(client, config, on_text_delta=on_text_delta)

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
    if config.daw:
        metadata["daw"] = config.daw
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


def append_language_to_prompt(user_prompt: str, language: Optional[str]) -> str:
    """Return the prompt combined with the detected language when available."""

    sanitized_prompt = user_prompt.strip()
    sanitized_language = language.strip() if isinstance(language, str) else ""

    if not sanitized_language:
        return sanitized_prompt

    if not sanitized_prompt:
        return f"[language: {sanitized_language}]"

    return f"[language: {sanitized_language}] {sanitized_prompt}"


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
    daw: Optional[str],
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
                daw=daw,
                agent_type="creative",
                conversation_id=conversation_id,
                previous_response_id=previous_response_id,
                prompt_reference=prompt_reference,
                cache_key=base,
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
            daw=daw,
            agent_type="app",
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            prompt_reference=prompt_reference,
            cache_key=base,
        ),
        True,
    )


def pcm_frames_to_wav(
    frames: bytes,
    sample_rate: int = 16000,
    *,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


def store_conversation_message(
    model: str,
    *,
    role: str,
    content: str = "",
    mode: str = "text",
    start: Optional[float] = None,
    end: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "role": role,
        "content": content or "",
        "mode": mode or "text",
    }
    if start is not None:
        message["start"] = start
    if end is not None:
        message["end"] = end
    for key, value in extra.items():
        if value is not None:
            message[key] = value
    history_state = st.session_state.setdefault("conversation_history", {})
    history = history_state.setdefault(model, [])
    history.append(message)
    return message


def append_text_history_message(model: str, role: str, content: str) -> Dict[str, Any]:
    timestamp = time.time()
    return store_conversation_message(
        model,
        role=role,
        content=content,
        mode="text",
        start=timestamp,
        end=timestamp,
    )


def format_timestamp(value: Optional[float]) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(value).strftime("%H:%M:%S")
    except (OverflowError, ValueError):
        return None


def _get_active_voice_turns() -> Dict[str, Any]:
    return st.session_state.setdefault("active_voice_turns", {})


def _register_voice_alias(store: Dict[str, Any], alias: str) -> None:
    if not alias:
        return
    active = _get_active_voice_turns()
    aliases = store.setdefault("aliases", set())
    if alias in aliases:
        return
    aliases.add(alias)
    active[alias] = store


def _resolve_voice_store(identifier: Optional[str]) -> Optional[Dict[str, Any]]:
    if not identifier:
        return None
    active = _get_active_voice_turns()
    store = active.get(identifier)
    if store:
        return store
    for candidate in active.values():
        if not isinstance(candidate, dict):
            continue
        if candidate.get("session_id") == identifier or candidate.get("response_id") == identifier:
            return candidate
    return None


def start_voice_turn(
    model: str,
    role: str,
    *,
    session_id: Optional[str] = None,
    start: Optional[float] = None,
) -> str:
    session_identifier = session_id or f"session-{uuid.uuid4()}"
    start_ts = start if start is not None else time.time()
    entry = store_conversation_message(
        model,
        role=role,
        content="",
        mode="voice",
        start=start_ts,
        session_id=session_identifier,
    )
    store = {
        "model": model,
        "entry": entry,
        "transcript": "",
        "frames": bytearray(),
        "sample_rate": None,
        "session_id": session_identifier,
        "aliases": set(),
    }
    _register_voice_alias(store, session_identifier)
    return session_identifier


def set_voice_response_id(identifier: Optional[str], response_id: Optional[str]) -> None:
    if not identifier or not response_id:
        return
    store = _resolve_voice_store(identifier)
    if not store:
        return
    store["response_id"] = response_id
    store["entry"]["response_id"] = response_id
    _register_voice_alias(store, response_id)


def append_voice_transcript(identifier: Optional[str], delta_text: Optional[str]) -> None:
    if not identifier or not delta_text:
        return
    store = _resolve_voice_store(identifier)
    if not store:
        return
    transcript = store.get("transcript", "") + delta_text
    store["transcript"] = transcript
    store["entry"]["content"] = transcript


def append_voice_audio_chunk(
    identifier: Optional[str],
    encoded_chunk: Optional[str],
    *,
    sample_rate: Optional[int] = None,
) -> None:
    if not identifier or not encoded_chunk:
        return
    store = _resolve_voice_store(identifier)
    if not store:
        return
    try:
        chunk_bytes = base64.b64decode(encoded_chunk)
    except (binascii.Error, ValueError):
        return
    frames: bytearray = store.setdefault("frames", bytearray())
    frames.extend(chunk_bytes)
    if sample_rate:
        store["sample_rate"] = sample_rate


def finalize_voice_turn(
    identifier: Optional[str],
    *,
    response_id: Optional[str] = None,
    session_id: Optional[str] = None,
    end: Optional[float] = None,
    sample_width: int = 2,
    channels: int = 1,
) -> Optional[Dict[str, Any]]:
    store = _resolve_voice_store(identifier)
    if not store:
        return None
    active = _get_active_voice_turns()
    for alias in list(store.get("aliases", set())):
        active.pop(alias, None)
    entry = store.get("entry", {})
    if response_id:
        entry["response_id"] = response_id
        store["response_id"] = response_id
    if session_id:
        entry["session_id"] = session_id
    entry["end"] = end if end is not None else time.time()
    transcript = store.get("transcript", "").strip()
    if transcript:
        entry["content"] = transcript
    frames = store.get("frames")
    sample_rate = store.get("sample_rate") or 16000
    if frames:
        audio_bytes = pcm_frames_to_wav(
            bytes(frames),
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )
        entry["audio"] = audio_bytes
        entry["audio_format"] = "audio/wav"
        audio_key = entry.get("audio_key")
        if not audio_key:
            base_key = entry.get("session_id") or identifier or str(uuid.uuid4())
            suffix = response_id or str(uuid.uuid4())
            audio_key = f"{base_key}:{suffix}"
            entry["audio_key"] = audio_key
        st.session_state.setdefault("voice_recordings", {})[audio_key] = audio_bytes
    return entry


def process_realtime_delta_event(event: Mapping[str, Any]) -> None:
    event_type = event.get("type") if isinstance(event, Mapping) else None
    if event_type == "response.output_text.delta":
        response = event.get("response") or {}
        response_id = response.get("id") or event.get("response_id")
        session = response.get("session") or {}
        session_id = session.get("id") or response.get("session_id") or event.get("session_id")
        delta = event.get("delta")
        delta_text: Optional[str] = None
        if isinstance(delta, str):
            delta_text = delta
        elif isinstance(delta, Mapping):
            text_value = delta.get("text")
            if isinstance(text_value, str):
                delta_text = text_value
        identifier = response_id or session_id
        if identifier:
            append_voice_transcript(identifier, delta_text)
    elif event_type == "response.output_audio.delta":
        response = event.get("response") or {}
        response_id = response.get("id") or event.get("response_id")
        session = response.get("session") or {}
        session_id = session.get("id") or response.get("session_id") or event.get("session_id")
        delta = event.get("delta") or {}
        encoded = delta.get("audio") or delta.get("pcm")
        sample_rate = delta.get("sample_rate")
        identifier = response_id or session_id
        if identifier:
            append_voice_audio_chunk(identifier, encoded, sample_rate=sample_rate)
    elif event_type == "response.created":
        response = event.get("response") or {}
        response_id = response.get("id") or event.get("response_id")
        session = response.get("session") or {}
        session_id = session.get("id") or response.get("session_id")
        identifier = session_id or response_id
        if identifier and response_id:
            set_voice_response_id(identifier, response_id)


def process_realtime_completed_event(event: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(event, Mapping):
        return None
    response = event.get("response") or {}
    response_id = response.get("id") or event.get("response_id")
    session = response.get("session") or {}
    session_id = session.get("id") or response.get("session_id") or event.get("session_id")
    identifier = response_id or session_id
    if not identifier:
        return None
    return finalize_voice_turn(
        identifier,
        response_id=response_id,
        session_id=session_id,
        end=time.time(),
    )


def attach_realtime_event_listeners(data_channel: Any) -> None:
    if data_channel is None or not hasattr(data_channel, "on"):
        return

    def _handle_message(message: Any) -> None:
        payload: Any = message
        if isinstance(message, bytes):
            try:
                payload = message.decode("utf-8")
            except UnicodeDecodeError:
                return
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return
        if not isinstance(payload, Mapping):
            return
        queue = st.session_state.setdefault("realtime_event_queue", [])
        queue.append(payload)
        event_type = payload.get("type") if isinstance(payload, Mapping) else None
        if event_type == "response.completed":
            entry = process_realtime_completed_event(payload)
            if entry is not None:
                payload["_entry"] = entry
        elif event_type in {
            "response.output_text.delta",
            "response.output_audio.delta",
            "response.created",
        }:
            process_realtime_delta_event(payload)
            payload["_processed_delta"] = True

    try:
        data_channel.on("message", _handle_message)
    except Exception:  # pylint: disable=broad-except
        return


def run_voice_session(
    client: OpenAI,  # pylint: disable=unused-argument
    config: RunConfig,
    *,
    on_text_delta: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], str]:
    session_identifier = config.voice_session_id
    if session_identifier:
        store = _resolve_voice_store(session_identifier)
        if not store:
            session_identifier = start_voice_turn(
                config.model,
                config.voice_role,
                session_id=session_identifier,
            )
    else:
        session_identifier = start_voice_turn(config.model, config.voice_role)

    queue = st.session_state.get("realtime_event_queue", [])
    response_payload: Dict[str, Any] = {}
    transcript = ""
    processed_indices: List[int] = []
    completed_event: Optional[Mapping[str, Any]] = None

    for index, event in enumerate(list(queue)):
        if not isinstance(event, Mapping):
            processed_indices.append(index)
            continue
        event_type = event.get("type")
        if event_type in {
            "response.output_text.delta",
            "response.output_audio.delta",
            "response.created",
        }:
            if not event.get("_processed_delta"):
                process_realtime_delta_event(event)
                event["_processed_delta"] = True
            if event_type == "response.output_text.delta":
                store = _resolve_voice_store(session_identifier)
                if store:
                    transcript = store.get("transcript", transcript)
                    if on_text_delta:
                        on_text_delta(transcript)
            processed_indices.append(index)
        elif event_type == "response.completed":
            completed_event = event
            processed_indices.append(index)

    if processed_indices:
        for index in reversed(processed_indices):
            queue.pop(index)

    if completed_event:
        response_payload = completed_event.get("response") or {}
        entry = completed_event.get("_entry")
        if entry is None:
            entry = process_realtime_completed_event(completed_event)
        if entry:
            transcript = entry.get("content", transcript)
        if session_identifier:
            response_payload.setdefault("session", {})["id"] = session_identifier
        if on_text_delta and transcript:
            on_text_delta(transcript)
        return response_payload, transcript.strip()

    store = _resolve_voice_store(session_identifier)
    if store:
        transcript = store.get("transcript", transcript)
        response_id = store.get("response_id")
        response_payload = {"session": {"id": session_identifier}}
        if response_id:
            response_payload["id"] = response_id
    return response_payload, transcript.strip()


def render_conversation_history(model: str) -> bool:
    history: List[Dict[str, Any]] = st.session_state["conversation_history"].get(
        model, []
    )
    if not history:
        return False

    for message in history:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role", "assistant")
        content = message.get("content", "")
        mode = message.get("mode", "text")
        start_label = format_timestamp(message.get("start"))
        end_label = format_timestamp(message.get("end"))
        metadata_tokens: List[str] = []
        if start_label:
            metadata_tokens.append(f"start {start_label}")
        if end_label:
            metadata_tokens.append(f"end {end_label}")
        session_id = message.get("session_id")
        response_id = message.get("response_id")
        audio_bytes = message.get("audio")
        audio_key = message.get("audio_key")
        if not audio_bytes and audio_key:
            audio_bytes = st.session_state.get("voice_recordings", {}).get(audio_key)
        with st.chat_message("user" if role == "user" else "assistant"):
            if mode == "voice":
                label = ":microphone: Voice message"
                if metadata_tokens:
                    label = f"{label} ({' • '.join(metadata_tokens)})"
                st.markdown(f"**{label}**")
                if content:
                    st.markdown(content)
                else:
                    st.markdown("_Transcription pending…_")
                if audio_bytes:
                    st.audio(audio_bytes, format=message.get("audio_format", "audio/wav"))
                meta_lines: List[str] = []
                if session_id:
                    meta_lines.append(f"Session `{session_id}`")
                if response_id:
                    meta_lines.append(f"Response `{response_id}`")
                if meta_lines:
                    st.caption(" • ".join(meta_lines))
            else:
                display_content = content if content else "_No content available._"
                st.markdown(display_content)

    return True


def extract_conversation_id(response: Dict[str, Any]) -> Optional[str]:
    session = response.get("session")
    if isinstance(session, dict):
        session_id = session.get("id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()

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
    st.set_page_config(page_title="Studio Pro Assistant", layout="wide")
    st.title("Studio Pro Assistant")
    st.caption("version 4.0.5 (251006)")

    developer_mode = is_developer_mode_enabled()
    realtime_config = get_realtime_config()
    stored_realtime = st.session_state.get("realtime_config")
    if isinstance(stored_realtime, Mapping):
        stored_model = stored_realtime.get("model")
        if isinstance(stored_model, str) and stored_model.strip():
            realtime_config["model"] = stored_model.strip()
        stored_voice = stored_realtime.get("voice")
        if isinstance(stored_voice, str) and stored_voice.strip():
            realtime_config["voice"] = stored_voice.strip()
        stored_latency = stored_realtime.get("latency_mode")
        if isinstance(stored_latency, str) and stored_latency.strip():
            realtime_config["latency_mode"] = stored_latency.strip()
        stored_rate = stored_realtime.get("sample_rate_hz")
        if isinstance(stored_rate, int) and stored_rate > 0:
            realtime_config["sample_rate_hz"] = stored_rate
        stored_ice = stored_realtime.get("ice_servers")
        if isinstance(stored_ice, Sequence) and not isinstance(
            stored_ice, (bytes, bytearray, str)
        ):
            realtime_config["ice_servers"] = [
                entry.strip()
                for entry in stored_ice
                if isinstance(entry, str) and entry.strip()
            ]

    if "realtime_config" not in st.session_state:
        st.session_state["realtime_config"] = dict(realtime_config)

    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
    if "voice_session" not in st.session_state:
        st.session_state["voice_session"] = None
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = {}
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = None

    if "active_voice_turns" not in st.session_state:
        st.session_state["active_voice_turns"] = {}
    if "voice_recordings" not in st.session_state:
        st.session_state["voice_recordings"] = {}
    if "realtime_event_queue" not in st.session_state:
        st.session_state["realtime_event_queue"] = []
    if "voice_session_toggle" not in st.session_state:
        st.session_state["voice_session_toggle"] = False
    if "voice_session_active" not in st.session_state:
        st.session_state["voice_session_active"] = False
    if "voice_session_started" not in st.session_state:
        st.session_state["voice_session_started"] = False
    if "voice_session_metadata" not in st.session_state:
        st.session_state["voice_session_metadata"] = None
    if "voice_session_error" not in st.session_state:
        st.session_state["voice_session_error"] = None

    # Update the `daw_options` list to support additional DAW versions in the future.
    daw_options = [
        "PreSonus Studio Pro 7",
    ]
    default_daw = st.session_state.get("selected_daw_version")
    active_model = st.session_state.get("active_model")
    if isinstance(active_model, str):
        active_state = st.session_state["conversations"].get(active_model, {})
        stored_daw = active_state.get("daw") if isinstance(active_state, Mapping) else None
        if isinstance(stored_daw, str) and stored_daw.strip():
            default_daw = stored_daw.strip()
    if not isinstance(default_daw, str) or default_daw not in daw_options:
        default_daw = daw_options[0]
    daw_version = st.selectbox(
        "DAW-Version for AI-context",
        options=daw_options,
        index=daw_options.index(default_daw),
        help="Add more DAW versions by editing the `daw_options` list.",
    )
    st.session_state["selected_daw_version"] = daw_version

    if developer_mode:
        st.divider()
        st.subheader("Realtime configuration")
        st.caption("These settings are only visible in developer mode.")

        realtime_model = st.text_input(
            "Realtime model",
            value=realtime_config["model"],
            help="Name of the realtime model to request when starting a voice session.",
        )

        voice_options = sorted(
            {realtime_config["voice"], "aria", "alloy", "verse"}
        )
        voice_index = 0
        if realtime_config["voice"] in voice_options:
            voice_index = voice_options.index(realtime_config["voice"])
        realtime_voice = st.selectbox(
            "Voice",
            options=voice_options,
            index=voice_index,
            help="Realtime voice preset used for audio synthesis.",
        )

        latency_options = ["default", "low_latency"]
        if realtime_config["latency_mode"] not in latency_options:
            latency_options.insert(0, realtime_config["latency_mode"])
        latency_index = 0
        if realtime_config["latency_mode"] in latency_options:
            latency_index = latency_options.index(realtime_config["latency_mode"])
        realtime_latency = st.selectbox(
            "Latency mode",
            options=latency_options,
            index=latency_index,
            help="Choose low_latency to prioritize minimal delay when supported.",
        )

        realtime_sample_rate = st.number_input(
            "Sample rate (Hz)",
            min_value=8000,
            step=1000,
            value=int(realtime_config["sample_rate_hz"]),
            help="Audio sample rate used for microphone capture and playback.",
        )

        ice_servers_text = st.text_area(
            "ICE servers",
            value="\n".join(realtime_config["ice_servers"]),
            help="Provide STUN/TURN URLs, one per line or comma separated.",
        )
        realtime_ice_servers = [
            entry.strip()
            for chunk in ice_servers_text.replace("\r", "\n").split("\n")
            for entry in chunk.split(",")
            if entry.strip()
        ]

        realtime_config["model"] = realtime_model.strip()
        realtime_config["voice"] = realtime_voice.strip()
        realtime_config["latency_mode"] = realtime_latency.strip()
        realtime_config["sample_rate_hz"] = int(realtime_sample_rate)
        realtime_config["ice_servers"] = realtime_ice_servers

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

    if "voice_session_toggle" not in st.session_state:
        st.session_state["voice_session_toggle"] = False

    st.markdown("### Voice session")
    voice_toggle = st.toggle(
        "Voice session",
        key="voice_session_toggle",
        help="Create a realtime session that captures microphone audio and plays the assistant's reply.",
    )
    controls_start, controls_stop = st.columns(2)
    start_clicked = controls_start.button(
        "Start",
        use_container_width=True,
        disabled=(
            not voice_toggle
            or st.session_state.get("voice_session_started", False)
            or st.session_state.get("voice_session_error") is not None
        ),
    )
    stop_clicked = controls_stop.button(
        "Stop",
        use_container_width=True,
        disabled=not st.session_state.get("voice_session_started", False),
    )

    if voice_toggle and not st.session_state.get("voice_session_active"):
        try:
            metadata = create_voice_session(client)
        except Exception as error:  # pylint: disable=broad-except
            st.session_state["voice_session_error"] = str(error)
            st.error(f"Failed to prepare the voice session: {error}")
            voice_toggle = False
        else:
            st.session_state["voice_session_metadata"] = metadata
            st.session_state["voice_session_active"] = True
            st.session_state["voice_session_error"] = None
    elif not voice_toggle and st.session_state.get("voice_session_active"):
        stop_voice_session()

    if start_clicked and st.session_state.get("voice_session_active"):
        st.session_state["voice_session_started"] = True
        st.session_state["voice_session_client"] = client
        st.session_state["voice_session_error"] = None

    if stop_clicked and st.session_state.get("voice_session_started"):
        stop_voice_stream()

    if st.session_state.get("voice_session_error"):
        st.error(st.session_state["voice_session_error"])
    elif st.session_state.get("voice_session_active") and not st.session_state.get(
        "voice_session_started"
    ):
        st.caption(
            "Enable the voice session and press Start. Your browser will prompt for microphone permissions—allow access to speak with the assistant."
        )

    if st.session_state.get("voice_session_started") and st.session_state.get(
        "voice_session_metadata"
    ):
        metadata = st.session_state["voice_session_metadata"]
        ice_servers = list(metadata.ice_servers) if metadata.ice_servers else [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
        rtc_configuration = RTCConfiguration(
            {"iceServers": ice_servers, "clientSecret": metadata.client_secret}
        )
        webrtc_ctx = webrtc_streamer(
            key="voice-session",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=VoiceAudioProcessor,
            async_processing=True,
        )
        st.session_state["voice_session_client"] = client
        if not webrtc_ctx.state.playing:
            st.info(
                "Click Start and allow microphone access in your browser to begin the realtime conversation."
            )

    voice_session_running = st.session_state.get("voice_session_started", False)

    run_button = st.button(
        "Generate responses",
        type="primary",
        disabled=voice_session_running,
    )

    user_prompt = prompt.strip()
    response_container = st.container()

    if run_button:
        if not user_prompt:
            st.warning("Please enter a question before generating a response.")
            st.stop()

        validation_error = validate_realtime_config(realtime_config)
        if validation_error:
            st.error(validation_error)
            st.stop()

        st.session_state["realtime_config"] = realtime_config

        conversations: Dict[str, Any] = st.session_state["conversations"]
        active_model = st.session_state.get("active_model")
        conversation_state: Dict[str, Any] = {}
        metadata_token: Optional[str] = None
        metadata_language: Optional[str] = None
        metadata_daw: str = daw_version.strip()
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
                stored_daw = state_entry.get("daw")
                if isinstance(stored_daw, str) and stored_daw.strip():
                    metadata_daw = stored_daw.strip()
                    st.session_state["selected_daw_version"] = metadata_daw
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
        else:
            # Preserve the DAW selection captured when the conversation started.
            st.session_state["selected_daw_version"] = metadata_daw

        metadata_token = metadata_token or resolve_metadata_token(route, None)

        prompt_config_name = "prompt_app" if route == "route_1" else "prompt_creative"
        prompt_config = get_prompt_config(prompt_config_name)
        prompt_input = user_prompt
        if not continuing_conversation:
            prompt_input = f"I am using {metadata_daw}. {prompt_input}" if prompt_input else f"I am using {metadata_daw}."
        prompt_for_model = append_language_to_prompt(prompt_input, metadata_language)
        prompt_reference = build_prompt_reference(
            prompt_config_name, prompt_for_model, config=prompt_config
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
            prompt_for_model,
            developer_prompt,
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            prompt_reference=prompt_reference,
            cache_key_base=cache_key_base,
            token=metadata_token,
            language=metadata_language,
            daw=metadata_daw,
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

        session_info = response.get("session")
        session_identifier: Optional[str] = None
        if isinstance(session_info, Mapping):
            session_raw = session_info.get("id")
            if isinstance(session_raw, str) and session_raw.strip():
                session_identifier = session_raw.strip()
        if not session_identifier and isinstance(existing_state, Mapping):
            existing_session = existing_state.get("session_id")
            if isinstance(existing_session, str) and existing_session.strip():
                session_identifier = existing_session.strip()
        new_state: Dict[str, Any] = {
            "conversation_id": conversation_identifier
            or existing_state.get("conversation_id"),
            "previous_response_id": response.get("id")
            or existing_state.get("previous_response_id"),
            "route": route,
            "token": metadata_token or existing_state.get("token"),
            "language": metadata_language or existing_state.get("language"),
            "daw": metadata_daw or existing_state.get("daw"),
            "session_id": session_identifier,
        }
        st.session_state["conversations"][target_model] = new_state

        append_text_history_message(target_model, "user", user_prompt)
        assistant_content = (
            sanitized_output
            if sanitized_output
            else "_No output text was returned._"
        )
        append_text_history_message(target_model, "assistant", assistant_content)

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
