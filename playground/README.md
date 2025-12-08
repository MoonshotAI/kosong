# Kosong Playground

A web-based playground for exploring how Kosong `Message` objects convert to different LLM provider formats.

## Features

- **JSON Input Editor**: Monaco-based editor with syntax highlighting
- **6 Example Messages**: Quick-start templates for common message types
- **5 Provider Outputs**: See conversions for Anthropic, OpenAI (Legacy & Responses), Google GenAI, and Kimi
- **Real-time Conversion**: Instant feedback on message format differences

## Quick Start

### Local Development

You need two terminal windows:

**Terminal 1 - Python API Server:**

```bash
cd playground
uv run python scripts/dev-api.py
```

**Terminal 2 - Next.js Dev Server:**

```bash
cd playground
pnpm dev
```

Open http://localhost:3000 in your browser.

### How It Works

1. Select an example or write your own Message JSON in the left panel
2. Click **Convert**
3. View the converted format for each provider in the right panel tabs

## Example Messages

| Example | Description |
|---------|-------------|
| Simple User Message | Basic text message |
| System Message | System prompt for assistant behavior |
| Message with Image | User message containing an ImageURLPart |
| Assistant with Tool Call | Assistant requesting a function call |
| Tool Response | Result from a tool call |
| Message with Thinking | Assistant with ThinkPart (reasoning) |

## Project Structure

```
playground/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── playground.tsx      # Main UI component
│   │   ├── query-provider.tsx  # React Query provider
│   │   └── ui/                 # shadcn/ui components
│   └── lib/
│       ├── types.ts            # TypeScript types for Message
│       ├── examples.ts         # Example messages
│       └── api.ts              # API client
├── api/
│   └── convert.py              # Vercel Python Serverless Function
├── scripts/
│   └── dev-api.py              # Local Python dev server
├── vercel.json                 # Vercel deployment config
├── next.config.ts              # Next.js config with API proxy
└── requirements.txt            # Python dependencies
```

## Tech Stack

- **Frontend**: Next.js 16, React 19, Tailwind CSS, shadcn/ui
- **Editor**: Monaco Editor
- **State Management**: TanStack Query (React Query)
- **Backend**: Vercel Python Functions
- **Converter**: kosong.conversion module

## Deployment

### Vercel

The project is configured for Vercel deployment. Dependencies (including `kosong` from PyPI) are declared in `requirements.txt`.

```bash
cd playground
vercel
```

### Requirements

- Node.js 18+
- Python 3.12+ (3.13 recommended)
- pnpm

## API Reference

### POST /api/convert

Convert a Kosong Message to all provider formats.

**Request:**

```json
{
  "message": {
    "role": "user",
    "content": "Hello, world!"
  }
}
```

**Response:**

```json
{
  "success": true,
  "results": {
    "anthropic": { "data": { ... } },
    "openai_legacy": { "data": { ... } },
    "openai_responses": { "data": [ ... ] },
    "google_genai": { "data": { ... } },
    "kimi": { "data": { ... } }
  },
  "input_validated": { ... }
}
```

## Related

- [kosong](../README.md) - The LLM abstraction layer
- [kosong.conversion](../src/kosong/conversion.py) - Standalone conversion functions
