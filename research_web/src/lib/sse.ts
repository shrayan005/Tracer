// Server-Sent Events, read by hand. The browser's EventSource can only GET (no body), and asking needs a POST,
// so the answer stream is read with fetch(): its body arrives in chunks, and a chunk can end mid-event.
//
//   event: stage            <- one event: "field: value" lines,
//   data: {"step":"reading"}
//                           <- ended by a blank line

export type SseEvent = { event: string; data: string };

/** Split what has arrived so far into whole events, and the unfinished tail to keep for the next chunk. */
export function parseSse(buffer: string): { events: SseEvent[]; rest: string } {
  const blocks = buffer.replace(/\r\n?/g, "\n").split("\n\n");
  const rest = blocks.pop() ?? ""; // after the last blank line: not complete yet
  const events: SseEvent[] = [];
  for (const block of blocks) {
    let event = "message"; // the SSE default when a block has no `event:` line
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // a comment (servers send these as keep-alives)
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
      if (field === "event") event = value;
      else if (field === "data") data.push(value); // several data lines join with newlines
    }
    if (data.length) events.push({ event, data: data.join("\n") });
  }
  return { events, rest };
}

/** Read a streaming response, calling `onEvent` for each event the moment it is complete. */
export async function readSse(response: Response, onEvent: (e: SseEvent) => void): Promise<void> {
  if (!response.body) throw new Error("this browser can't read a streamed response");
  const reader = response.body.getReader();
  const decoder = new TextDecoder(); // stream: true keeps a character split across two chunks intact
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
    const { events, rest } = parseSse(done ? buffer + "\n\n" : buffer);
    events.forEach(onEvent);
    buffer = rest;
    if (done) return;
  }
}
