"use client";

import { useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import Editor from "@monaco-editor/react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { convertMessage } from "@/lib/api";
import { examples } from "@/lib/examples";
import type { ConvertResponse, ProviderName, Message } from "@/lib/types";

const PROVIDERS: { id: ProviderName; name: string }[] = [
  { id: "anthropic", name: "Anthropic" },
  { id: "openai_legacy", name: "OpenAI Legacy" },
  { id: "openai_responses", name: "OpenAI Responses" },
  { id: "google_genai", name: "Google GenAI" },
  { id: "kimi", name: "Kimi" },
];

type RequestPreview = {
  method?: string;
  url?: string;
  path?: string;
  headers?: Record<string, string>;
  body?: unknown;
  raw?: unknown;
};

type ProviderRenderResult =
  | { type: "error"; content: string }
  | { type: "success"; request: RequestPreview };

const parseRequestPreview = (httpRequest: unknown): RequestPreview => {
  const req = (httpRequest ?? {}) as Record<string, unknown>;
  const url = typeof req.url === "string" ? req.url : undefined;
  let path = url;
  try {
    if (url) {
      const u = new URL(url);
      path = `${u.pathname}${u.search}`;
    }
  } catch {
    path = url;
  }

  const headers =
    req.headers && typeof req.headers === "object"
      ? (req.headers as Record<string, string>)
      : undefined;

  const body =
    "json" in req
      ? (req.json as unknown)
      : "body" in req
        ? (req.body as unknown)
        : "data" in req
          ? (req.data as unknown)
          : undefined;

  return {
    method: typeof req.method === "string" ? req.method : undefined,
    url,
    path,
    headers,
    body,
    raw: httpRequest,
  };
};

const formatBody = (body: unknown) => {
  if (body === undefined || body === null) return "None";
  if (typeof body === "string") return body;
  try {
    return JSON.stringify(body, null, 2);
  } catch {
    return String(body);
  }
};

export function Playground() {
  const [input, setInput] = useState(() =>
    JSON.stringify(examples[0].message, null, 2)
  );
  const [parseError, setParseError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: async (messageJson: string) => {
      const parsed = JSON.parse(messageJson) as Message;
      return convertMessage(parsed);
    },
  });

  const handleConvert = useCallback(() => {
    setParseError(null);
    try {
      JSON.parse(input);
      mutation.mutate(input);
    } catch (e) {
      setParseError(e instanceof Error ? e.message : "Invalid JSON");
    }
  }, [input, mutation]);

  const handleExampleChange = useCallback((value: string) => {
    const example = examples.find((e) => e.name === value);
    if (example) {
      setInput(JSON.stringify(example.message, null, 2));
      setParseError(null);
    }
  }, []);

  const getProviderResult = (
    results: ConvertResponse["results"] | undefined,
    provider: ProviderName
  ): ProviderRenderResult | null => {
    if (!results) return null;
    const result = results[provider];
    if (!result) return null;
    if (result.error) {
      return { type: "error" as const, content: result.error };
    }
    const request = parseRequestPreview(result.http_request ?? result);
    return { type: "success", request };
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <header className="border-b bg-white dark:bg-zinc-900">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Kosong Playground</h1>
          <Select onValueChange={handleExampleChange}>
            <SelectTrigger className="w-[220px]">
              <SelectValue placeholder="Select an example..." />
            </SelectTrigger>
            <SelectContent>
              {examples.map((example) => (
                <SelectItem key={example.name} value={example.name}>
                  {example.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Section */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Message JSON Input</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border rounded-md overflow-hidden h-[400px]">
                <Editor
                  height="100%"
                  defaultLanguage="json"
                  value={input}
                  onChange={(value) => setInput(value || "")}
                  theme="vs-dark"
                  options={{
                    minimap: { enabled: false },
                    fontSize: 13,
                    lineNumbers: "on",
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    tabSize: 2,
                  }}
                />
              </div>
              {parseError && (
                <div className="text-sm text-red-500 bg-red-50 dark:bg-red-950 p-2 rounded">
                  {parseError}
                </div>
              )}
              <Button
                onClick={handleConvert}
                disabled={mutation.isPending}
                className="w-full"
              >
                {mutation.isPending ? "Converting..." : "Convert"}
              </Button>
              {mutation.error && (
                <div className="text-sm text-red-500 bg-red-50 dark:bg-red-950 p-2 rounded">
                  {mutation.error instanceof Error
                    ? mutation.error.message
                    : "Conversion failed"}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Output Section */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Provider Outputs</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="anthropic" className="w-full">
                <TabsList className="grid w-full grid-cols-5">
                  {PROVIDERS.map((provider) => {
                    const result = getProviderResult(
                      mutation.data?.results,
                      provider.id
                    );
                    return (
                      <TabsTrigger
                        key={provider.id}
                        value={provider.id}
                        className="text-xs relative"
                      >
                        {provider.name.split(" ")[0]}
                        {result && (
                          <span
                            className={`absolute -top-1 -right-1 w-2 h-2 rounded-full ${
                              result.type === "success"
                                ? "bg-green-500"
                                : "bg-red-500"
                            }`}
                          />
                        )}
                      </TabsTrigger>
                    );
                  })}
                </TabsList>
                {PROVIDERS.map((provider) => {
                  const result = getProviderResult(
                    mutation.data?.results,
                    provider.id
                  );
                  return (
                    <TabsContent
                      key={provider.id}
                      value={provider.id}
                      className="mt-4"
                    >
                  <div className="border rounded-md overflow-hidden h-[350px] bg-zinc-950 text-zinc-100">
                    {!result && (
                      <div className="h-full flex items-center justify-center text-zinc-400">
                        Click &quot;Convert&quot; to see the output
                      </div>
                    )}
                    {result?.type === "error" && (
                      <div className="h-full p-4 text-sm text-red-300 bg-red-950">
                        {result.content}
                      </div>
                    )}
                    {result?.type === "success" && (
                      <div className="h-full p-4 overflow-auto space-y-4 text-sm">
                        <div className="space-y-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="px-2 py-1 rounded bg-emerald-600/80 text-xs font-semibold uppercase">
                              {result.request.method || "REQ"}
                            </span>
                            <code className="text-xs break-all">
                              {result.request.url || "Unknown URL"}
                            </code>
                          </div>
                          <div className="text-xs text-zinc-300">
                            <span className="font-semibold text-zinc-200">
                              Path:
                            </span>
                            <code className="ml-2 break-all">
                              {result.request.path || "—"}
                            </code>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <div className="text-xs font-semibold text-zinc-200">
                            Headers
                          </div>
                          {result.request.headers &&
                          Object.keys(result.request.headers).length > 0 ? (
                            <div className="space-y-1 font-mono text-[11px]">
                              {Object.entries(result.request.headers).map(
                                ([key, value]) => (
                                  <div
                                    key={key}
                                    className="flex gap-2 flex-wrap"
                                  >
                                    <span className="text-sky-300">{key}:</span>
                                    <span className="break-all text-zinc-100">
                                      {String(value)}
                                    </span>
                                  </div>
                                )
                              )}
                            </div>
                          ) : (
                            <div className="text-xs text-zinc-400">None</div>
                          )}
                        </div>

                        <div className="space-y-2">
                          <div className="text-xs font-semibold text-zinc-200">
                            Body
                          </div>
                          <pre className="bg-zinc-900 border border-zinc-800 rounded p-2 text-[11px] whitespace-pre-wrap break-words overflow-auto">
                            {formatBody(
                              result.request.body ?? result.request.raw
                            )}
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                    </TabsContent>
                  );
                })}
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
