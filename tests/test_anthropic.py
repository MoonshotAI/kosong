# from inline_snapshot import snapshot

# from kosong.contrib.chat_provider.anthropic import message_to_anthropic
# from kosong.message import Message, ToolCall


# def test_message_to_anthropic_includes_tool_use_block_for_string_content() -> None:
#     message = Message(
#         role="assistant",
#         content="6",
#         tool_calls=[
#             ToolCall(
#                 id="abc",
#                 function=ToolCall.FunctionBody(
#                     name="foo",
#                     arguments='{"x":1}',
#                 ),
#             )
#         ],
#     )

#     anthropic_payload = message_to_anthropic(message)

#     assert anthropic_payload == snapshot(
#         {
#             "role": "assistant",
#             "content": [
#                 {"type": "text", "text": "6"},
#                 {"type": "tool_use", "id": "abc", "name": "foo", "input": {"x": 1}},
#             ],
#         }
#     )

# FIXME: test this by capturing requests sent to a mock http server
