// The topics open as tabs, remembered in this browser (localStorage) so they survive a refresh.
// Only a convenience: if storage is blocked (private window), tabs simply aren't remembered.
import { useCallback, useSyncExternalStore } from "react";

export type OpenTopic = { id: number; title: string };
const KEY = "research.openTopics";
const MAX = 8;
const listeners = new Set<() => void>();
let cache: OpenTopic[] = read();

function read(): OpenTopic[] {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(value) ? value.filter((t) => typeof t?.id === "number").slice(0, MAX) : [];
  } catch {
    return [];
  }
}

function write(next: OpenTopic[]) {
  cache = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* storage blocked: keep them in memory only */
  }
  listeners.forEach((notify) => notify());
}

export function useOpenTopics() {
  const topics = useSyncExternalStore(
    (notify) => { listeners.add(notify); return () => listeners.delete(notify); },
    () => cache,
  );
  const open = useCallback((topic: OpenTopic) => {
    const known = cache.find((t) => t.id === topic.id);
    if (known && known.title === topic.title) return;
    write(known ? cache.map((t) => (t.id === topic.id ? topic : t)) : [...cache, topic].slice(-MAX));
  }, []);
  const close = useCallback((id: number) => write(cache.filter((t) => t.id !== id)), []);
  const clear = useCallback(() => write([]), []);
  return { topics, open, close, clear };
}
