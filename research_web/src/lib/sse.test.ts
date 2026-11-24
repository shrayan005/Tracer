import { describe, expect, it } from "vitest";
import { parseSse, readSse } from "./sse";

describe("parseSse", () => {
  it("returns whole events and keeps an unfinished one for the next chunk", () => {
    const { events, rest } = parseSse('event: stage\ndata: {"step":"searching"}\n\nevent: stage\ndata: {"st');
    expect(events).toEqual([{ event: "stage", data: '{"step":"searching"}' }]);
    expect(rest).toBe('event: stage\ndata: {"st');
  });

  it("handles \\r\\n line ends, comments, the default event name and multi-line data", () => {
    const { events } = parseSse(": keep-alive\r\n\r\ndata: a\r\ndata: b\r\n\r\n");
    expect(events).toEqual([{ event: "message", data: "a\nb" }]);
  });
});

describe("readSse", () => {
  it("puts events back together across chunk borders, even inside a character", async () => {
    const bytes = new TextEncoder().encode('event: done\ndata: {"q":"≥5 g"}\n\n');
    const cut = bytes.indexOf(0xe2) + 1; // split the 3-byte "≥" in the middle
    const body = new ReadableStream<Uint8Array>({
      start(c) { c.enqueue(bytes.slice(0, cut)); c.enqueue(bytes.slice(cut)); c.close(); },
    });
    const got: string[] = [];
    await readSse(new Response(body), (e) => got.push(`${e.event} ${e.data}`));
    expect(got).toEqual(['done {"q":"≥5 g"}']);
  });

  it("reads a last event even when the stream ends without the blank line", async () => {
    const got: string[] = [];
    await readSse(new Response("event: error\ndata: {}"), (e) => got.push(e.event));
    expect(got).toEqual(["error"]);
  });
});
