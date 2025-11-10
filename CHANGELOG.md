# Changelog

## [0.23.0] - 2025-11-10

- Change type of `ToolError.output` to `str | ContentPart | Sequence[ContentPart]`.

## [0.22.0] - 2025-11-10

- Add `APIEmptyResponseError` for cases where the API returns an empty response.
- Add `GenerateResult` as the return type of `generate` function.
- Add `id: str | None` field to `GenerateResult` and `StepResult`.
