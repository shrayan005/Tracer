import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Logo } from "../components/Logo";
import { useLogin, useRegister } from "../api/queries";

export function LoginPage() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const login = useLogin();
  const register = useRegister();
  const navigate = useNavigate();
  const from = (useLocation().state as { from?: string } | null)?.from ?? "/app";

  const action = mode === "login" ? login : register;
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const done = { onSuccess: () => navigate(from, { replace: true }) };
    if (mode === "login") login.mutate({ email, password }, done);
    else register.mutate({ email, password }, done);
  };

  return (
    <div className="login">
      <form className="login-box" onSubmit={submit}>
        <Link to="/" className="login-brand"><Logo size={22} /></Link>
        <h1>{mode === "login" ? "Welcome back" : "Create your account"}</h1>
        <p className="muted login-sub">{mode === "login" ? "Log in to your research." : "Start tracing every claim to its source."}</p>
        {action.error && <p className="error">{action.error.message}</p>}
        <label className="field">
          <span>Email</span>
          <input className="input" type="email" autoComplete="email" required autoFocus
                 value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label className="field">
          <span>Password{mode === "register" && " (10 characters or more)"}</span>
          <input className="input" type="password" required minLength={mode === "register" ? 10 : undefined}
                 autoComplete={mode === "login" ? "current-password" : "new-password"}
                 value={password} onChange={(e) => setPassword(e.target.value)} />
        </label>
        <button className="button" disabled={action.isPending}>
          {action.isPending ? "…" : mode === "login" ? "Log in" : "Create account"}
        </button>
        <p className="muted" style={{ fontSize: 13, marginBottom: 0 }}>
          {mode === "login" ? "No account? " : "Have an account? "}
          <button type="button" className="link-button" onClick={() => { setMode(mode === "login" ? "register" : "login"); action.reset(); }}>
            {mode === "login" ? "Create one" : "Log in"}
          </button>
        </p>
      </form>
    </div>
  );
}
