// TypeScript types matching Kosong Message structure

export interface TextPart {
  type: "text";
  text: string;
}

export interface ThinkPart {
  type: "think";
  think: string;
  encrypted?: string | null;
}

export interface ImageURLPart {
  type: "image_url";
  image_url: {
    url: string;
    id?: string | null;
  };
}

export interface AudioURLPart {
  type: "audio_url";
  audio_url: {
    url: string;
    id?: string | null;
  };
}

export type ContentPart = TextPart | ThinkPart | ImageURLPart | AudioURLPart;

export interface FunctionBody {
  name: string;
  arguments: string | null;
}

export interface ToolCall {
  type: "function";
  id: string;
  function: FunctionBody;
  extras?: Record<string, unknown> | null;
}

export type Role = "system" | "user" | "assistant" | "tool";

export interface Message {
  role: Role;
  name?: string | null;
  content: ContentPart[] | string;
  tool_calls?: ToolCall[] | null;
  tool_call_id?: string | null;
  partial?: boolean | null;
}

export type ProviderName =
  | "anthropic"
  | "openai_legacy"
  | "openai_responses"
  | "google_genai"
  | "kimi";

export interface ProviderResult {
  http_request?: unknown;
  error?: string;
}

export interface ConvertResponse {
  success: boolean;
  results: Record<ProviderName, ProviderResult>;
  input_validated?: Message;
  error?: string;
}
