# import json
import base64
import io
import shlex
from pathlib import Path

from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image as FastMCPImage
from mcp.types import TextContent

from pm_env.tools.bash import bash

MAX_IMG_DIM = 2000


# Should we resize images before uploading? Right now claude api errors
# if model image is too large.
#
# Claude tokens = (width px * height px) / 750
#
# From claude docs
# "Claude works best when images come before text" - interesting. We are
# not submitting images + text at the same time ever (for now).
#
# TODO Should we allow viewing multiple image files at once?
#
# Note that if the image filename extension is incorrect, the api call will fail.
async def view_image_file(file_path: Path) -> ToolResult:
    """
    Views the image files at absolute path file_path.

    The image must be a .jpeg, .png, .gif, or .webp and must be less than 5MB.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError(
            "Pillow is required to use the view_image_file tool. "
            + "Install it with `uv add Pillow`."
        )

    if not file_path.is_absolute():
        raise ValueError(f"File path must be absolute: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use bash tool to get async execution and correct permission handling
    # TODO how low can we make max_output_length?
    bash_tool = bash(max_output_length=10000000)

    try:
        escaped_path = shlex.quote(str(file_path))

        result: ToolResult = await bash_tool(command=f"base64 -w 0 {escaped_path}")
        assert isinstance(result.content[0], TextContent)

        if result.structured_content is not None:
            if "truncated" in result.structured_content.get("system", ""):
                return ToolResult(
                    structured_content={
                        "result": "Image is too large. Please try a smaller image."
                    }
                )
            image_b64_bytes = base64.b64decode(
                result.structured_content.get("stdout", "")
            )
            img = PILImage.open(io.BytesIO(image_b64_bytes))

            if img.width > MAX_IMG_DIM or img.height > MAX_IMG_DIM:
                return ToolResult(
                    structured_content={
                        "result": f"Image is too large. Please try a smaller image. Width and height of image must be less than {MAX_IMG_DIM} pixels."
                    }
                )

            img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="jpeg")

            img_base64 = buffered.getvalue()

        else:
            print("ALERT setting image bytes to nothing")
            img_base64 = b""

        # Make MCP image to make it easy to format into content
        img = FastMCPImage(data=img_base64, format="jpeg")
        img_content = img.to_image_content()

        return ToolResult(content=[img_content])
    finally:
        await bash_tool.dispose()
