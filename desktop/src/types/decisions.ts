/**
 * Схема decisions.json — вынесена из старого TimelineScreen'а, чтобы
 * DashboardScreen / TimelineScreen / ExportScreen делили один тип.
 *
 * Соответствует backend'у: mutes/cuts в миллисекундах, highlights в секундах
 * (так исторически лёг NotebookLM-конвейер — не трогаем).
 */

export interface Mute {
  start_ms: number;
  end_ms: number;
  words?: string[];
  source?: string;
}

export interface Cut {
  start_ms: number;
  end_ms: number;
  reason?: string;
}

export interface Highlight {
  start_s: number;
  end_s: number;
  title: string;
  reason: string;
  score: number;
  quote?: string;
}

export interface HighlightsBundle {
  highlights?: Highlight[];
}

export interface Decisions {
  mutes?: Mute[];
  cuts?: Cut[];
  highlights?: HighlightsBundle;
}
