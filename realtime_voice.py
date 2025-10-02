"""Utilities for managing OpenAI real-time voice sessions."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import streamlit as st
from openai import OpenAI


LOGGER = logging.getLogger(__name__)

VOICE_SESSION_KEY = "voice_session"


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


@dataclass
class VoiceSession:
    """State for an OpenAI real-time voice session."""

    session_id: str
    ephemeral_api_key: str
    client_secret: str
    websocket_url: str
    model: str
    voice: str

    @property
    def is_ready(self) -> bool:
        """Return ``True`` when the session has the data needed for WebRTC."""

        return bool(self.session_id and self.client_secret and self.websocket_url)


def _ensure_dict(data: Any) -> Mapping[str, Any]:
    if hasattr(data, "model_dump"):
        return data.model_dump()  # type: ignore[no-any-return]
    if hasattr(data, "to_dict"):
        return data.to_dict()  # type: ignore[no-any-return]
    if isinstance(data, Mapping):
        return data
    return {}


def _extract_secret(payload: Any) -> str:
    mapping = _ensure_dict(payload)
    value = mapping.get("value")
    if isinstance(value, str):
        return value
    return ""


def _resolve_default(key: str, fallback: str) -> str:
    realtime = st.secrets.get("realtime", {})
    if isinstance(realtime, Mapping):
        for candidate in (key, f"default_{key}"):
            configured = realtime.get(candidate)
            if isinstance(configured, str) and configured.strip():
                return configured.strip()
    return fallback


def _normalize_preference(preferred: str, *, key: str, fallback: str) -> str:
    if isinstance(preferred, str) and preferred.strip():
        return preferred.strip()
    return _resolve_default(key, fallback)


def _store_session(session: Optional[VoiceSession]) -> None:
    st.session_state[VOICE_SESSION_KEY] = session


def get_voice_session() -> Optional[VoiceSession]:
    """Return the active voice session stored in ``st.session_state``."""

    session = st.session_state.get(VOICE_SESSION_KEY)
    return session if isinstance(session, VoiceSession) else None


def create_voice_session(client: OpenAI, model: str, voice: str) -> VoiceSession:
    """Create and persist a new real-time voice session."""

    resolved_model = _normalize_preference(
        model, key="model", fallback="gpt-4o-realtime-preview-2024-12-17"
    )
    resolved_voice = _normalize_preference(voice, key="voice", fallback="verse")

    sessions_resource = _get_realtime_sessions_resource(client)
    session = sessions_resource.create(model=resolved_model, voice=resolved_voice)
    session_dict = _ensure_dict(session)

    session_id = session_dict.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("Failed to obtain a valid session id from the real-time API response.")

    websocket_url = session_dict.get("websocket_url")
    if not isinstance(websocket_url, str) or not websocket_url:
        webrtc = _ensure_dict(session_dict.get("webrtc"))
        websocket_url = webrtc.get("url")
        if not isinstance(websocket_url, str) or not websocket_url:
            urls_section = _ensure_dict(session_dict.get("urls"))
            websocket_url = urls_section.get("websocket")
            if not isinstance(websocket_url, str):
                websocket_url = ""

    api_secret = _extract_secret(session_dict.get("client_secret"))
    webrtc_secret = _extract_secret(
        _ensure_dict(session_dict.get("webrtc")).get("client_secret")
    )

    voice_session = VoiceSession(
        session_id=session_id,
        ephemeral_api_key=api_secret,
        client_secret=webrtc_secret,
        websocket_url=websocket_url,
        model=session_dict.get("model") or resolved_model,
        voice=session_dict.get("voice") or resolved_voice,
    )
    _store_session(voice_session)
    return voice_session


def refresh_voice_session(
    client: OpenAI, *, model: Optional[str] = None, voice: Optional[str] = None
) -> VoiceSession:
    """Replace the current voice session with a new one."""

    current = get_voice_session()
    if current:
        try:
            sessions_resource = _get_realtime_sessions_resource(client)
            sessions_resource.delete(current.session_id)
        except Exception as exc:  # pragma: no cover - best effort clean-up
            LOGGER.debug("Failed to delete existing session during refresh: %s", exc)

    return create_voice_session(
        client,
        model or (current.model if current else ""),
        voice or (current.voice if current else ""),
    )


def end_voice_session(client: OpenAI) -> None:
    """Terminate and clear the active voice session."""

    current = get_voice_session()
    if not current:
        _store_session(None)
        return

    try:
        sessions_resource = _get_realtime_sessions_resource(client)
        sessions_resource.delete(current.session_id)
    except Exception as exc:  # pragma: no cover - best effort clean-up
        LOGGER.debug("Failed to delete voice session: %s", exc)
    finally:
        _store_session(None)
