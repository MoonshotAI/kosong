# Changelog

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
