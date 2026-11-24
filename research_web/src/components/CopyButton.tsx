import { Copy } from "lucide-react";
import { useToast } from "./Toast";

/** Copy a text (a citation) to the clipboard; a small toast confirms it. */
export function CopyButton({ text, label }: { text: string; label: string }) {
  const toast = useToast();
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      toast(`${label} copied`);
    } catch {
      window.prompt("Copy:", text);           // clipboard blocked (e.g. plain http on another host)
    }
  };
  return (
    <button className="chip" onClick={copy} title={`Copy ${label}`}>
      <Copy size={13} /> {label}
    </button>
  );
}
