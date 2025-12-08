import type { Message } from "./types";

export interface Example {
  name: string;
  description: string;
  message: Message;
}

export const examples: Example[] = [
  {
    name: "Simple User Message",
    description: "Basic user message with text content",
    message: {
      role: "user",
      content: "Hello, world!",
    },
  },
  {
    name: "System Message",
    description: "System prompt to set assistant behavior",
    message: {
      role: "system",
      content: "You are a helpful assistant that specializes in Python programming.",
    },
  },
  {
    name: "Message with Image",
    description: "User message containing an image URL",
    message: {
      role: "user",
      content: [
        { type: "text", text: "What's in this image?" },
        {
          type: "image_url",
          image_url: { url: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png" },
        },
      ],
    },
  },
  {
    name: "Assistant with Tool Call",
    description: "Assistant message requesting a tool call",
    message: {
      role: "assistant",
      content: "Let me calculate that for you.",
      tool_calls: [
        {
          type: "function",
          id: "call_abc123",
          function: {
            name: "calculator",
            arguments: JSON.stringify({ expression: "2 + 2" }),
          },
        },
      ],
    },
  },
  {
    name: "Tool Response",
    description: "Tool result message responding to a tool call",
    message: {
      role: "tool",
      content: "4",
      tool_call_id: "call_abc123",
    },
  },
  {
    name: "Message with Thinking",
    description: "Assistant message with thinking/reasoning content",
    message: {
      role: "assistant",
      content: [
        {
          type: "think",
          think: "Let me think about this step by step...\n1. First, I need to understand the problem.\n2. Then, I'll formulate my response.",
          encrypted: "signature_placeholder_abc123",
        },
        { type: "text", text: "The answer is 42." },
      ],
    },
  },
];
