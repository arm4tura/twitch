import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertCircle } from "lucide-react";

/**
 * ErrorBoundary — ловит throw из рендера дочернего дерева и показывает ошибку
 * вместо чёрного экрана. Без него любое исключение в рендере (например в
 * TimelineScreen) размонтирует весь root → пустое окно без единой подсказки.
 *
 * Показываем message + stack прямо в UI: это dev-инструмент, пользователю всё
 * равно нужно скинуть текст ошибки, а нам — увидеть её без DevTools.
 */

interface Props {
  children: ReactNode;
  /** Метка места — попадает в заголовок, чтобы отличать границы в дереве. */
  label?: string;
}

interface State {
  error: Error | null;
  info: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, info: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Дублируем в консоль — Electron DevTools/терминал увидят полный стек.
    console.error("[ErrorBoundary]", this.props.label ?? "", error, info);
    this.setState({ info });
  }

  handleReset = (): void => {
    this.setState({ error: null, info: null });
  };

  render(): ReactNode {
    const { error, info } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="mx-auto max-w-3xl overflow-auto p-8">
        <div className="rounded-lg border border-err/30 bg-err/5 p-6">
          <div className="flex items-center gap-2 text-err">
            <AlertCircle className="h-5 w-5" />
            <h2 className="text-base font-semibold">
              Экран упал{this.props.label ? ` — ${this.props.label}` : ""}
            </h2>
          </div>
          <p className="mt-3 font-mono text-sm text-fg">{error.message}</p>
          {(error.stack || info?.componentStack) && (
            <pre className="mt-4 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-black/40 p-3 text-[11px] leading-relaxed text-subtle">
              {error.stack}
              {info?.componentStack ? `\n\nComponent stack:${info.componentStack}` : ""}
            </pre>
          )}
          <button
            type="button"
            onClick={this.handleReset}
            className="mt-4 rounded-md border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-fg hover:bg-white/10"
          >
            Попробовать снова
          </button>
        </div>
      </div>
    );
  }
}
