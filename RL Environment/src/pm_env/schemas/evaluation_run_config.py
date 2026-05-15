from pydantic import BaseModel

from .http_mcp_server_config import HttpMcpServerConfig
from .websocket_config import WebSocketConfig

GCP_PROJECT = "deft-haven-445201-k3"
GCP_LOCATION = "global"  # gemini 3 pro is in global.


class EvaluationRunConfig(BaseModel):
    run_id: str
    task_id: str
    model: str
    model_api_key: str | None = None
    """API key for the model."""

    mcp_server_config: HttpMcpServerConfig = HttpMcpServerConfig()

    websocket_config: WebSocketConfig = WebSocketConfig()
    """Where to stream the run's progress."""

    transcript_file: str | None = None
    """Where to save the run's transcript."""
