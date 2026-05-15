from pydantic import BaseModel


class HttpMcpServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
