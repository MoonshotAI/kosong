# Changelog

## [Unreleased]

- Fix token usage for Anthropic chat provider

## [0.27.2] - 2025-11-25

- Support non-OpenAI models which do not accept `developer` role in system prompt in `OpenAIResponses` chat provider

## [0.27.1] - 2025-11-24

- Nothing

## [0.27.0] - 2025-11-24

- Fix function call ID in `GoogleGenAI` chat provider
- Make `CallableTool2` not a `pydantic.BaseModel`
- Introduce `ToolReturnValue` as the common base class of `ToolOk` and `ToolError`
- Require `CallableTool` and `CallableTool2` to return `ToolReturnValue` instead of `ToolOk | ToolError`
- Rename `ToolResult.result` to `ToolResult.return_value`

## [0.26.2] - 2025-11-20

- Better thinking level mapping in `GoogleGenAI` chat provider

## [0.26.1] - 2025-11-19

- Deref JSON schema in tool parameters to fix compatibility with some LLM providers

## [0.26.0] - 2025-11-19

- Fix thinking part in `Anthropic` provider's non-stream mode
- Add `GoogleGenAI` chat provider

## [0.25.1] - 2025-11-18

- Catch httpx exceptions correctly in Kimi and OpenAI providers

## [0.25.0] - 2025-11-13

- Add `reasoning_key` argument to `OpenAILegacy` chat provider to specify the field for reasoning content in messages

## [0.24.0] - 2025-11-12

- Set default temperature settings for Kimi models based on model name

## [0.23.0] - 2025-11-10

- Change type of `ToolError.output` to `str | ContentPart | Sequence[ContentPart]`

## [0.22.0] - 2025-11-10

- Add `APIEmptyResponseError` for cases where the API returns an empty response
- Add `GenerateResult` as the return type of `generate` function
- Add `id: str | None` field to `GenerateResult` and `StepResult`
