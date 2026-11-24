import { Component, type ReactNode } from "react";

type Props = { children: ReactNode; resetKey: string };
type State = { error: Error | null };

/** A crash in one page shows a message in that pane instead of blanking the whole app.
 *  Navigating elsewhere (a new resetKey) clears it. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(previous: Props) {
    if (previous.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="empty">
        <strong>Something went wrong on this page</strong>
        {this.state.error.message}
        <p><button className="small-button" onClick={() => this.setState({ error: null })}>Try again</button></p>
      </div>
    );
  }
}
