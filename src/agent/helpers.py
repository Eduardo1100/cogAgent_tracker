import ast
import copy
import re
from collections.abc import Callable

from autogen import ConversableAgent
from autogen.agentchat.contrib.capabilities import transform_messages
from sentence_transformers import SentenceTransformer, util


def parse_tool_call(tool_call_string: str) -> tuple[str, tuple]:
    """
    Parse a tool call string to extract the tool call name and parameters.
    For example: "function_name('param1', 42)" -> ("function_name", ("param1", 42))
    """
    pattern = r"(\w+)\((.*?)\)"
    match = re.match(pattern, tool_call_string)
    if match:
        tool_call_name = match.group(1)
        parameters = match.group(2).strip()
        if parameters:
            # Safely evaluate parameters
            try:
                # Wrap parameters so ast.literal_eval interprets them as a tuple
                args = ast.literal_eval(f"({parameters},)")
            except SyntaxError:
                # If something is malformed, treat parameters as a single string
                args = (parameters,)
        else:
            args = ()
        return tool_call_name, args
    else:
        raise ValueError(f"Invalid tool call string format: {tool_call_string}")


class MessageToolCall:
    def __init__(self, tool_dict: dict[str, Callable]):
        # Ensure all values in tool_dict are callable.
        self.tool_dict = tool_dict
        for _, tool in tool_dict.items():
            if not callable(tool):
                raise ValueError("All tools must be callable functions.")

    def _transform_text_content(self, text: str) -> str:
        """
        For a given text string, find and replace all occurrences of tool calls
        defined in self.tool_dict.
        """
        # For each tool_name, repeatedly find and replace all calls
        for tool_name, func in self.tool_dict.items():
            # Build a pattern that matches this specific tool call
            # Note: The non-greedy .*? is used to match minimal parameters
            # While still allowing multiple calls.
            pattern = rf"{re.escape(tool_name)}\((.*?)\)"

            match = re.search(pattern, text)
            if not match:
                # No more occurrences of this tool
                continue
            # Extract the full matched substring
            full_call_str = text[match.start() : match.end()]
            # Parse it
            parsed_tool_name, args = parse_tool_call(full_call_str)
            if parsed_tool_name == tool_name:
                result = func(*args)
                return f"ECHO: {result}"
            else:
                # If somehow parsing didn't match the tool_name,
                # break to avoid infinite loop
                raise ValueError(
                    f"Tool name mismatch: {parsed_tool_name} != {tool_name}"
                )
        return text

    def apply_transform(self, messages: list[dict]) -> list[dict]:
        temp_messages = copy.deepcopy(messages)

        message = temp_messages[-1]
        # If content is a simple string
        if isinstance(message["content"], str):
            message["content"] = self._transform_text_content(message["content"])

        # If content is a list, iterate over text-type items
        elif isinstance(message["content"], list):
            for item in message["content"]:
                if item.get("type") == "text" and isinstance(item["text"], str):
                    item["text"] = self._transform_text_content(item["text"])

        return temp_messages

    def get_logs(
        self, pre_transform_messages: list[dict], post_transform_messages: list[dict]
    ) -> tuple[str, bool]:
        # Compare pre and post transformation messages for changes.
        for message, post_message in zip(
            pre_transform_messages, post_transform_messages
        ):
            if message["content"] != post_message["content"]:
                return "Function call triggered", True
        return "", False


def register_function_lambda(
    tool_dict: dict[str, Callable], agents: list[ConversableAgent]
):
    tool_handling = transform_messages.TransformMessages(
        transforms=[MessageToolCall(tool_dict)]
    )
    for agent in agents:
        tool_handling.add_to_agent(agent)


sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_best_candidate(reference_sentence, candidate_sentences):
    # Compute embeddings
    target_embedding = sentence_transformer_model.encode(
        reference_sentence, convert_to_tensor=True
    )
    command_embeddings = sentence_transformer_model.encode(
        candidate_sentences, convert_to_tensor=True
    )

    # Compute cosine similarity
    similarities = util.cos_sim(target_embedding, command_embeddings)

    # Find the most similar command
    most_similar_idx = similarities.argmax()
    most_similar_command = candidate_sentences[most_similar_idx]
    score = similarities.detach().cpu().numpy()[0, most_similar_idx]

    return most_similar_command, score


def is_termination_msg_generic(msg):
    return any(
        keyword in (msg.get("content") or "") for keyword in ["FLEECE", "STRAWBERRY"]
    )


def get_echo_agent(name, llm_config, additional_termination_criteria=None):
    if additional_termination_criteria is None:
        additional_termination_criteria = []

    def termination_criteria(msg):
        criterion = is_termination_msg_generic(msg)
        return criterion or any(
            criteria(msg) for criteria in additional_termination_criteria
        )

    echo_agent = ConversableAgent(
        name=f"{name}" if name else "echo",
        system_message=f"You are {name}, if the last message you received begins with"
        'the keyword "ECHO: ", then you parrot the contents of the last message you'
        "received. Otherwise, do nothing."
        f"\nExample:"
        f"\n Last message you received = ECHO: Observation: You arrive at drawer 2. The"
        "drawer 2 is closed. Task Status: INCOMPLETE Actions Left: 20 Current"
        "Admissible Actions: ['examine drawer 2', 'go to bed 1', 'go to desk 1', 'go to"
        " drawer 1', 'go to drawer 3', 'go to drawer 4', 'go to drawer 5', 'go to "
        "dresser 1', 'go to garbagecan 1', 'go to laundryhamper 1', 'go to shelf 1', "
        "'help', 'inventory', 'look', 'open drawer 2']"
        f"\n Your output = Observation: You arrive at drawer 2. The drawer 2 is closed."
        " Task Status: INCOMPLETE Actions Left: 20 Current Admissible Actions:"
        " ['examine drawer 2', 'go to bed 1', 'go to desk 1', 'go to drawer 1', 'go "
        "to drawer 3', 'go to drawer 4', 'go to drawer 5', 'go to dresser 1', 'go to "
        "garbagecan 1', 'go to laundryhamper 1', 'go to shelf 1', 'help', 'inventory',"
        " 'look', 'open drawer 2']",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=termination_criteria,
    )
    echo_agent.description = "echoes the output of given function calls."
    return echo_agent


def create_echo_agent():
    """Creates a programmatic relay that turns tool outputs into standard text."""
    echo_agent = ConversableAgent(
        name="Echo_Agent",
        system_message="I am a relay. I convert environmental tool outputs into "
        "standard text.",
        llm_config=False,  # No LLM needed
        human_input_mode="NEVER",
    )

    # Track last known observation so we can repeat it if the relay misses
    _state = {"last_obs": None}

    def relay_observation(recipient, messages, sender, config):
        if not messages:
            return False, None

        # AutoGen appends an empty text reply from External_Perception_Agent AFTER
        # the tool result, so messages[-1] is never role='tool'. Search backwards
        # over a wider window to survive AutoGen version differences and any extra
        # housekeeping messages inserted between tool execution and this call.
        for msg in reversed(messages[-8:]):
            if msg.get("role") == "tool":
                content = msg.get("content") or "Action completed."
                _state["last_obs"] = content
                return True, f"[Observation]: {content}"

        # No tool message found — repeat the last known percept so agents never
        # drift into "awaiting observation" loops after history truncation.
        if _state["last_obs"] is not None:
            return True, f"[No new tool result — repeating last observation]: {_state['last_obs']}"
        return True, "[Internal State Synchronized]"

    echo_agent.register_reply([ConversableAgent, None], relay_observation)
    return echo_agent


class ConvertOrphanedToolMessages:
    """Fix broken tool_calls / tool pairings left by history truncation.

    Applied after MessageHistoryLimiter so that truncation never produces
    messages that strict OpenAI-compatible APIs reject with a 400.

    Two kinds of breakage can occur after the window is trimmed:

    1. role='tool' whose tool_call_id has no matching tool_calls in the
       window (the assistant tool_calls was cut off the front).
       Fix: convert the orphaned tool message to role='user'.

    2. role='assistant' with tool_calls whose required tool responses are
       missing (the tool message was never matched, e.g. the pairing was
       broken by intermediate non-tool messages after truncation).
       Fix: strip the tool_calls key so it becomes a plain assistant message
       (text noting the call is kept so Motor_Agent still sees an example).

    Unlike FlattenToolMessages this transform leaves intact all tool_calls /
    tool pairs that ARE correctly adjacent, so Motor_Agent keeps working
    examples of the tool-call format in its history.
    """

    def apply_transform(self, messages: list[dict]) -> list[dict]:
        msgs = copy.deepcopy(messages)

        # Pass 1 — find which assistant tool_calls messages are "complete":
        # all of their tool_call_ids have a matching role='tool' immediately
        # following (no non-tool messages in between).
        complete_tool_call_ids: set[str] = set()
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tc_ids = {tc["id"] for tc in msg.get("tool_calls", []) if "id" in tc}
                j = i + 1
                found: set[str] = set()
                while j < len(msgs) and msgs[j].get("role") == "tool":
                    tid = msgs[j].get("tool_call_id")
                    if tid:
                        found.add(tid)
                    j += 1
                if tc_ids and tc_ids == found:
                    complete_tool_call_ids |= tc_ids
            i += 1

        # Pass 2 — rewrite messages.
        result: list[dict] = []
        pending_ids: set[str] = set()

        for msg in msgs:
            role = msg.get("role")

            if role == "assistant" and "tool_calls" in msg:
                tc_ids = {tc["id"] for tc in msg.get("tool_calls", []) if "id" in tc}
                if tc_ids <= complete_tool_call_ids:
                    # All calls are complete — keep tool_calls intact.
                    pending_ids = tc_ids
                    result.append(msg)
                else:
                    # Incomplete — strip tool_calls so the API doesn't reject.
                    calls = [
                        f"[Calling {c['function']['name']}]"
                        for c in msg.get("tool_calls", [])
                    ]
                    msg["content"] = (msg.get("content") or "") + "\n" + "\n".join(calls)
                    msg.pop("tool_calls", None)
                    pending_ids = set()
                    result.append(msg)

            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and tid in pending_ids:
                    pending_ids.discard(tid)
                    result.append(msg)
                else:
                    # Orphaned tool response — convert to plain user observation.
                    content = msg.get("content") or "No observation provided."
                    msg["role"] = "user"
                    msg["content"] = f"[Observation]: {content}"
                    msg.pop("tool_call_id", None)
                    msg.pop("tool_responses", None)
                    result.append(msg)

            else:
                pending_ids = set()
                result.append(msg)

        return result

    def get_logs(
        self, pre_transform_messages: list[dict], post_transform_messages: list[dict]
    ) -> tuple[str, bool]:
        changed = any(
            pre != post
            for pre, post in zip(pre_transform_messages, post_transform_messages)
        )
        return ("Orphaned tool messages converted" if changed else ""), changed


class FlattenToolMessages:
    """TransformMessages-compatible transform that strips AutoGen's tool protocol.

    Converts ``role='tool'`` messages to ``role='user'`` observations and removes
    ``tool_calls`` keys from assistant messages so DeepSeek Reasoner (and other
    strict OpenAI-compat APIs) don't reject the conversation history with a 400.
    """

    def apply_transform(self, messages: list[dict]) -> list[dict]:
        return flatten_tool_messages(messages)

    def get_logs(
        self, pre_transform_messages: list[dict], post_transform_messages: list[dict]
    ) -> tuple[str, bool]:
        changed = any(
            pre != post
            for pre, post in zip(pre_transform_messages, post_transform_messages)
        )
        return ("Tool protocol flattened" if changed else ""), changed


def flatten_tool_messages(messages):
    if not messages:
        return []

    scrubbed = []
    for msg in copy.deepcopy(messages):
        # 1. Convert 'tool' responses to 'user' observations.
        # Also strip 'tool_responses' — ag2's _generate_oai_reply_from_client
        # unrolls that key back into role='tool' entries, re-triggering the 400.
        if msg.get("role") == "tool":
            msg["role"] = "user"
            content = msg.get("content") or "No observation provided."
            msg["content"] = f"[Observation]: {content}"
            msg.pop("tool_call_id", None)
            msg.pop("tool_responses", None)

        # 2. Strip 'tool_calls' from assistant messages so Reasoner doesn't 400
        if "tool_calls" in msg:
            calls = [
                f"[Calling {c['function']['name']}]" for c in msg.get("tool_calls", [])
            ]
            msg["content"] = (msg.get("content") or "") + "\n" + "\n".join(calls)
            msg.pop("tool_calls", None)

        scrubbed.append(msg)
    return scrubbed
