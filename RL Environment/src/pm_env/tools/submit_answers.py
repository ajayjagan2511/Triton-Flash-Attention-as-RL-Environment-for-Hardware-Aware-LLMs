from fastmcp.tools.tool import ToolResult


def submit_answers(
    answers: dict[str, str]
) -> ToolResult:  # pyright: ignore[reportUnusedParameter]
    """Each entry in `answers` is one answer. Can be called multiple times to add more answers.

    Example for the question "What is the capital of France? How big is the population?":
        submit_answers({"capital": "Paris", "population": "11.3 million"})
    """
    # This is a placeholder function to register the tool with the MCP server.
    # However, any calls to this tool get intercepted and processed by the `EvaluationRunner`
    # instead of the MCP server. The evaluation runner takes the submitted answers and
    # adds them directly to `transcript.answers`. Check out `_submit_answers` in
    # `evaluation_runner.py` for the details.
    ...
