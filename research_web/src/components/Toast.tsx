import { Check } from "lucide-react";
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";

type Show = (message: string) => void;
const ToastContext = createContext<Show>(() => {});

/** Tiny confirmations ("Citation copied") at the bottom of the screen; one at a time, gone after 2 s. */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [message, setMessage] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const show = useCallback<Show>((text) => {
    setMessage(text);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setMessage(null), 2000);
  }, []);
  return (
    <ToastContext.Provider value={show}>
      {children}
      <div className="toast-area" role="status" aria-live="polite">
        {message && <div className="toast"><Check size={14} /> {message}</div>}
      </div>
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);
