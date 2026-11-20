"""
Client factory for routing model requests to the correct backend.

Supports four routing paths, ALL using AsyncOpenAI under the hood:
  1a. Gemini (Vertex AI, service account key.json) — via OpenAI-compatible endpoint
  1b. Gemini (AI Studio, simple API key) — via OpenAI-compatible endpoint
  2.  External API (OpenAI, etc.) — native OpenAI API
  3.  Local vLLM server — OpenAI-compatible API

All backends return an object implementing the same interface:
    client.chat.completions.create(model, messages, **kwargs) -> ChatCompletion

Architecture note (2025-02 migration):
    Previously, Route 1a used a custom GeminiOAuthClient class (~200 lines) that
    wrapped the Vertex AI SDK, manually converted message formats, and constructed
    ChatCompletion objects by hand.  The new design uses Vertex AI's OpenAI-compatible
    endpoint (https://aiplatform.googleapis.com/.../endpoints/openapi), eliminating
    all custom conversion code.  The only Vertex-specific logic is OAuth token refresh.

Usage:
    from rllm.utils.client_factory import create_client, is_external_model

    # Gemini via Vertex AI (service account)
    client = create_client("gemini-2.5-pro", key_file="key.json")

    # Gemini via AI Studio (API key)
    client = create_client("gemini-2.5-pro", api_key="AIza...")

    # OpenAI API
    client = create_client("gpt-4o", base_url="https://api.openai.com/v1")

    # Local vLLM server
    client = create_client("Qwen/Qwen3-4B", base_url="http://localhost:30000/v1")

    # All return the same interface:
    response = await client.chat.completions.create(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )
    print(response.choices[0].message.content)
"""

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# =============================================================================
# External model detection
# =============================================================================

# Model prefixes that require external API (not local vLLM)
EXTERNAL_MODEL_PREFIXES = ("gemini", "gpt-", "claude-", "o1-", "o3-")


def is_external_model(model_name: str) -> bool:
    """Check if a model requires an external API (not a local vLLM server).

    Args:
        model_name: Model identifier string.

    Returns:
        True if the model is served by an external API (Gemini, OpenAI, etc.),
        False if it should be accessed via a local vLLM server.
    """
    return any(model_name.lower().startswith(p) for p in EXTERNAL_MODEL_PREFIXES)


# =============================================================================
# VertexAIOpenAIClient — AsyncOpenAI wrapper with OAuth token auto-refresh
# =============================================================================


class VertexAIOpenAIClient:
    """AsyncOpenAI wrapper that authenticates to Vertex AI's OpenAI-compatible endpoint.

    Uses a Google Cloud service account key.json to obtain OAuth2 tokens and
    automatically refreshes them before they expire (~1 hour lifetime).

    The Vertex AI OpenAI-compatible endpoint is:
        https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi

    Model names on this endpoint require a "google/" prefix (e.g., "google/gemini-2.5-pro").
    This wrapper auto-prefixes model names that don't already have "google/".

    Args:
        key_file: Path to Google Cloud service account key.json.
        project_id: Google Cloud project ID. If None, auto-detected from key.json.
        location: Google Cloud region (default: "global").
                  Note: Gemini 3 models are ONLY available on "global".
                  Gemini 2.5 models work on both "global" and regional endpoints.
    """

    def __init__(self, key_file: str, project_id: str = None, location: str = "global"):
        # Lazy import to avoid hard dependency on google-auth
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as exc:
            raise ImportError(
                "google-auth package not found. Install with:\n  pip install google-auth google-auth-oauthlib\nThis replaces the old google-cloud-aiplatform/vertexai dependency."
            ) from exc

        self._google_auth_requests = google.auth.transport.requests

        # Load credentials — supports multiple key.json formats:
        #   - "service_account"  (from GCP Console → IAM → Service Accounts)
        #   - "authorized_user"  (from gcloud auth application-default login)
        #   - "external_account" (workload identity federation)
        key_path = os.path.abspath(key_file)
        with open(key_path) as f:
            key_data = json.load(f)

        cred_type = key_data.get("type", "unknown")
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]

        if cred_type == "service_account":
            from google.oauth2 import service_account

            self._credentials = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=scopes,
            )
        elif cred_type == "authorized_user":
            from google.oauth2 import credentials as user_credentials

            self._credentials = user_credentials.Credentials(
                token=key_data.get("access_token"),
                refresh_token=key_data.get("refresh_token"),
                token_uri=key_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=key_data.get("client_id"),
                client_secret=key_data.get("client_secret"),
                scopes=scopes,
            )
        else:
            # Fallback: set env var and let google.auth.default() figure it out
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
            self._credentials, detected_project = google.auth.default(scopes=scopes)
            if not project_id and detected_project:
                project_id = detected_project

        logger.info(f"Loaded credentials: type={cred_type}")

        # Auto-detect project_id from key.json if not provided
        if not project_id:
            project_id = key_data.get("quota_project_id") or key_data.get("project_id") or getattr(self._credentials, "project_id", None)
        if not project_id:
            raise ValueError(f"Could not determine project_id from {key_file} (type={cred_type}). Please provide --project_id explicitly.")

        self.project_id = project_id
        self.location = location

        # Build Vertex AI OpenAI-compatible endpoint URL
        base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi"

        # Get initial OAuth token
        self._refresh_token()

        # Create the underlying AsyncOpenAI client
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=self._credentials.token,
        )

        logger.info(f"VertexAIOpenAIClient initialized: project={project_id}, location={location}, base_url={base_url}")

        # Expose for compatibility with code that checks these attributes
        self.base_url = self._client.base_url
        self.api_key = self._credentials.token

    def _refresh_token(self):
        """Refresh OAuth token if expired or about to expire."""
        if not self._credentials.valid:
            self._credentials.refresh(self._google_auth_requests.Request())
            # Update the client's API key with the new token
            if hasattr(self, "_client"):
                self._client.api_key = self._credentials.token
                self.api_key = self._credentials.token
            logger.debug("Vertex AI OAuth token refreshed")

    # ---- Chat namespace with auto-prefix + token refresh ----

    class _ChatNamespace:
        """Wraps AsyncOpenAI.chat to auto-prefix model names with 'google/'."""

        def __init__(self, vertex_client: "VertexAIOpenAIClient"):
            self._vertex_client = vertex_client
            self.completions = self._Completions(vertex_client)

        class _Completions:
            def __init__(self, vertex_client: "VertexAIOpenAIClient"):
                self._vertex_client = vertex_client

            async def create(self, **kwargs):
                # Auto-prefix model name for Vertex AI endpoint
                model = kwargs.get("model", "")
                if model and not model.startswith("google/"):
                    kwargs["model"] = f"google/{model}"
                return await self._vertex_client._client.chat.completions.create(**kwargs)

    @property
    def chat(self):
        """Auto-refresh token before returning chat namespace with model auto-prefix."""
        self._refresh_token()
        return self._ChatNamespace(self)

    @property
    def completions(self):
        """Auto-refresh token before returning completions namespace."""
        self._refresh_token()
        return self._client.completions

    def __repr__(self):
        return f"VertexAIOpenAIClient(project_id={self.project_id!r}, location={self.location!r})"


# =============================================================================
# GeminiAIStudioWrapper — AsyncOpenAI wrapper with thinking support
# =============================================================================


class GeminiAIStudioWrapper:
    """Thin wrapper around AsyncOpenAI that auto-injects thinking config.

    Works with BOTH Gemini endpoints:
      - AI Studio:  https://generativelanguage.googleapis.com/v1beta/openai/
      - Vertex AI:  https://aiplatform.googleapis.com/.../endpoints/openapi

    Both endpoints accept the same extra_body format for thinking config.

    When include_thoughts=True, every chat.completions.create() call
    automatically adds Google's thinking config to extra_body, so the
    response includes <thought>...</thought> tags in the content.

    When include_thoughts=False (default), this is a transparent passthrough.
    """

    def __init__(self, client, include_thoughts: bool = False):
        self._client = client
        self.include_thoughts = include_thoughts
        # Expose common attributes for compatibility
        self.base_url = getattr(client, "base_url", None)
        self.api_key = getattr(client, "api_key", None)

    class _ChatNamespace:
        def __init__(self, wrapper: "GeminiAIStudioWrapper"):
            self.completions = self._Completions(wrapper)

        class _Completions:
            def __init__(self, wrapper: "GeminiAIStudioWrapper"):
                self._wrapper = wrapper

            async def create(self, **kwargs):
                if self._wrapper.include_thoughts:
                    # Inject thinking config into extra_body for include_thoughts
                    # Cannot use reasoning_effort when thinking_config is present
                    extra_body = kwargs.pop("extra_body", {}) or {}
                    google_config = extra_body.setdefault("google", {})
                    thinking_config = google_config.setdefault("thinking_config", {})
                    thinking_config["include_thoughts"] = True

                    # Use thinking_budget for Gemini 2.5 (thinking_level is for Gemini 3)
                    # Note: thinking_budget can be buggy, but it's the only option for 2.5
                    # Gemini 2.5 Pro: min=512, max=24576
                    # Control max response length with thinking_budget, refere to: https://ai.google.dev/gemini-api/docs/thinking
                    if "thinking_budget" not in thinking_config:
                        thinking_config["thinking_budget"] = 2048  # Low budget
                    kwargs["extra_body"] = extra_body
                    # print(f"[DEBUG GEMINI] extra_body={kwargs.get('extra_body')}")
                else:
                    # When not including thoughts, use reasoning_effort (top-level, more reliable)
                    if "reasoning_effort" not in kwargs:
                        kwargs["reasoning_effort"] = "low"
                    # print(f"[DEBUG GEMINI] reasoning_effort={kwargs.get('reasoning_effort')}")
                return await self._wrapper._client.chat.completions.create(**kwargs)

    @property
    def chat(self):
        return self._ChatNamespace(self)

    @property
    def completions(self):
        return self._client.completions

    def __repr__(self):
        return f"GeminiAIStudioWrapper(include_thoughts={self.include_thoughts}, base_url={self.base_url!r})"


# =============================================================================
# Client factory
# =============================================================================


def create_client(
    model: str,
    base_url: str = None,
    api_key: str = None,
    key_file: str = None,
    project_id: str = None,
    location: str = "global",
    include_thoughts: bool = True,
) -> Any:
    """Create a client for the given model, routing to the correct backend.

    All four routes return an AsyncOpenAI-compatible client:

        Route 1a: "gemini-*" + key_file  → VertexAIOpenAIClient (OAuth, auto-refresh)
        Route 1b: "gemini-*" + api_key   → AsyncOpenAI (AI Studio endpoint)
        Route 2:  "gpt-*", "o1-*", etc.  → AsyncOpenAI (OpenAI API)
        Route 3:  everything else         → AsyncOpenAI (local vLLM server)

    For Gemini routes (1a, 1b), if include_thoughts=True, the client is
    additionally wrapped in GeminiAIStudioWrapper which auto-injects thinking
    config so responses include <thought>...</thought> tags.

    API key rotation:
        If GEMINI_API_KEYS env var is set (comma-separated), multiple keys are
        loaded and automatically rotated on 429 (RPD exhausted) errors.

    Args:
        model: Model name/identifier (e.g., "gemini-2.5-pro", "Qwen/Qwen3-4B").
        base_url: API endpoint URL. Required for vLLM and optional for others.
        api_key: API key string. For Gemini, this can be a Google AI Studio key
                 (starting with "AIza"). If None, reads from environment variable.
        key_file: Path to Google Cloud service account key.json (Gemini Vertex AI).
        project_id: Google Cloud project ID (Vertex AI only, auto-detected from
                     key.json if not provided).
        location: Google Cloud region (default: "global").
                  Gemini 3 models require "global". Gemini 2.5 works on both.
        include_thoughts: If True, Gemini thinking models return their internal
                          reasoning in <thought>...</thought> tags (debug only).

    Returns:
        Client object implementing client.chat.completions.create().

    Raises:
        ImportError: If required packages for Vertex AI auth are not installed.
        ValueError: If required configuration is missing.
    """
    model_lower = model.lower()

    # ---- Route 1: Gemini ----
    if model_lower.startswith("gemini") or model_lower.startswith("google/gemini"):
        # Route 1a: Vertex AI OpenAI-compatible endpoint (service account key.json)
        if key_file:
            # Also set env var for any code that reads it directly
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(key_file)
            logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS={key_file}")

            client = VertexAIOpenAIClient(
                key_file=key_file,
                project_id=project_id,
                location=location,
            )

            if include_thoughts:
                client = GeminiAIStudioWrapper(client, include_thoughts=True)
                logger.info(f"Created VertexAIOpenAIClient + GeminiAIStudioWrapper for {model} (include_thoughts=True)")
            else:
                logger.info(f"Created VertexAIOpenAIClient for {model}")

            return client

        # Route 1b: Google AI Studio (simple API key, OpenAI-compatible endpoint)
        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if resolved_api_key:
            gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            client = AsyncOpenAI(base_url=gemini_base_url, api_key=resolved_api_key)
            if include_thoughts:
                client = GeminiAIStudioWrapper(client, include_thoughts=True)
                logger.info(f"Created GeminiAIStudioWrapper for {model} (include_thoughts=True)")
            else:
                logger.info(f"Created AsyncOpenAI client for {model} via Google AI Studio")
            return client

        raise ValueError(
            f"Gemini model '{model}' requires authentication. Provide one of:\n"
            f"  --key-file key.json  (Vertex AI service account)\n"
            f"  --api-key AIza...    (Google AI Studio API key)\n"
            f"  Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable"
        )

    # ---- Route 2: OpenAI API ----
    if model_lower.startswith(("gpt-", "o1-", "o3-")):
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base_url = base_url or "https://api.openai.com/v1"
        if not resolved_api_key:
            raise ValueError(f"API key required for {model}. Set OPENAI_API_KEY or pass api_key parameter.")
        client = AsyncOpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
        logger.info(f"Created AsyncOpenAI client for {model}")
        return client

    # ---- Route 3: Local vLLM server ----
    resolved_base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:30000/v1")
    resolved_api_key = api_key or "EMPTY"
    client = AsyncOpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
    logger.info(f"Created AsyncOpenAI client for {model} at {resolved_base_url}")
    return client


def default_external_base_url(model: str) -> str:
    """OpenAI-compatible endpoint for an external model family (plain-API-key
    routes only; the Vertex OAuth route ignores base_url entirely).

    Callers routing an external role/policy model MUST NOT fall back to the
    local vLLM URL — that silently sends e.g. gemini-2.5-pro requests to the
    checkpoint server (2026-08-11 audit finding).
    """
    name = (model or "").lower()
    if name.startswith("gemini"):
        return "https://generativelanguage.googleapis.com/v1beta/openai/"
    if name.startswith("claude-"):
        return "https://api.anthropic.com/v1/"
    return "https://api.openai.com/v1"
