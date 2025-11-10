# Changelog

## [Unreleased]

- Add `APIEmptyResponseError` for cases where the API returns an empty response.
- Add `GenerateResult` as the return type of `generate` function.
- Add `id: str | None` field to `GenerateResult` and `StepResult`.
- Change type of `ToolError.output` to `str | ContentPart | Sequence[ContentPart]`.
