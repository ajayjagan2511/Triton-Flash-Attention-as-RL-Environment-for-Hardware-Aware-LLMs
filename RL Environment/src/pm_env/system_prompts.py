from pm_env.schemas.chat import Message

system_prompt = """You have access to a set of functions you can use to answer the user's question. This includes access to a sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external resources, except by invoking the below functions.

You can invoke one or more functions by writing a "<antml:function_calls>" block like the following as part of your reply to the user:
<antml:function_calls>
<antml:invoke name="$FUNCTION_NAME">
<antml:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</antml:parameter>
...
</antml:invoke>
<antml:invoke name="$FUNCTION_NAME2">
...
</antml:invoke>
</antml:function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.

The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of your reply to the user.
You may then continue composing the rest of your reply to the user, respond to any errors, or make further function calls as appropriate.
If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not recognized as a call."""

# Additional prompt for OpenAI models to ensure autonomous behavior.
# In practice we saw this model stopping halfway to confirm with the user.
_OPENAI_AUTONOMOUS_PROMPT = """

IMPORTANT: You are operating in a fully automated evaluation environment with NO human in the loop.
- NEVER ask for permission or confirmation before proceeding
- NEVER use phrases like "If you want...", "Should I...", "Would you like me to...", "Let me know if..."
- Always make decisions autonomously and proceed with your best judgment
- If you encounter an error, fix it yourself and continue - do not ask what to do
- Complete the task fully without waiting for input or feedback
- Do not explain what you "could" do - just do it"""

# System prompt for models that support native tool calling (like Kimi).
# These models receive tools via the API and should use structured tool_calls in responses.
# We explicitly tell them NOT to use XML format since that causes hallucination issues.
_NATIVE_TOOL_CALLING_PROMPT = """You have access to a set of functions you can use to answer the user's question. This includes access to a sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external resources, except by invoking the provided functions.

Use the tools provided to you via the API to complete the task. When you need to call a function, use the native tool calling format - do NOT write XML tags like <function_calls> in your response.

After each tool call, you will receive the result. You may then continue with your task, respond to any errors, or make further tool calls as appropriate.

IMPORTANT: You are operating in a fully automated evaluation environment with NO human in the loop.
- NEVER ask for permission or confirmation before proceeding
- NEVER use phrases like "If you want...", "Should I...", "Would you like me to...", "Let me know if..."
- Always make decisions autonomously and proceed with your best judgment
- If you encounter an error, fix it yourself and continue - do not ask what to do
- Complete the task fully without waiting for input or feedback
- Do not explain what you "could" do - just do it"""

# Models that should use native tool calling instead of XML format
_NATIVE_TOOL_CALLING_MODEL_PREFIXES = [
    "together_ai/moonshotai/Kimi",
]


def _uses_native_tool_calling(model: str) -> bool:
    """Check if a model should use native tool calling instead of XML format."""
    return any(
        model.startswith(prefix) for prefix in _NATIVE_TOOL_CALLING_MODEL_PREFIXES
    )


def get_system_message(model: str) -> Message:
    """Returns the appropriate system message based on the model.

    OpenAI models (GPT) receive additional instructions to ensure autonomous behavior,
    as they tend to ask for user confirmation before proceeding with actions.

    Models in _NATIVE_TOOL_CALLING_MODEL_PREFIXES receive a different prompt that instructs them
    to use native tool calling instead of XML format, which prevents hallucination issues.
    """
    if _uses_native_tool_calling(model):
        content = _NATIVE_TOOL_CALLING_PROMPT
    elif model.startswith("openai/") or model.startswith("gpt"):
        content = system_prompt + _OPENAI_AUTONOMOUS_PROMPT
    else:
        content = system_prompt
    return Message(
        content=content,
        role="system",
        tool_calls=None,
        reasoning_content=None,
        tool_call_id=None,
    )
