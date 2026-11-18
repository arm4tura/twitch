import { useEffect, useMemo, useRef, useState } from "react";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";
import { fmtMs } from "../../lib/format";
import { Button } from "../ui/Button";

export interface ReviewItem {
  id: string;
  start: number;
  end: number;
  label: string;
  detail: string;
  selected: boolean;
  onSelect: () => void;
}

/** Bounded rendering keeps a long stream's review queue inexpensive. */
export function ReviewList({ items, layer }: { items: ReviewItem[]; layer: string }) {
  const rowsRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const filtered = useMemo(() => items.filter((item) => `${item.label} ${item.detail}`.toLocaleLowerCase().includes(query.toLocaleLowerCase())), [items, query]);
  const selectedId = items.find((item) => item.selected)?.id;
  const selectedIndex = filtered.findIndex((item) => item.id === selectedId);
  const pages = Math.max(1, Math.ceil(filtered.length / 50));
  const current = Math.min(page, pages - 1);
  useEffect(() => { setPage(0); }, [layer, query]);
  useEffect(() => { if (selectedIndex >= 0) setPage(Math.floor(selectedIndex / 50)); }, [selectedId, selectedIndex]);
  useEffect(() => { rowsRef.current?.querySelector('[aria-pressed="true"]')?.scrollIntoView({ block: "nearest" }); }, [selectedId, current]);
  return (
    <aside className="review-list" aria-label="Фрагменты для проверки">
      <div className="review-list-heading"><strong>Фрагменты</strong><span>{items.length}</span></div>
      <label className="review-search"><Search size={14} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Поиск фрагмента…" aria-label="Поиск фрагмента" /></label>
      <div className="review-rows" ref={rowsRef}>
        {filtered.slice(current * 50, (current + 1) * 50).map((item) => (
          <button type="button" key={item.id} aria-pressed={item.selected} className={`review-row review-${layer}`} onClick={item.onSelect}>
            <span className="font-mono-tabular text-[10px] text-muted">{fmtMs(item.start * 1000)} — {fmtMs(item.end * 1000)}</span>
            <strong>{item.label}</strong><span className="text-xs text-muted">{item.detail}</span>
          </button>
        ))}
        {!filtered.length && <p className="p-4 text-xs text-muted">{query ? "Ничего не найдено" : "В этом слое пока нет фрагментов"}</p>}
      </div>
      {pages > 1 && <div className="flex items-center justify-between border-t border-white/10 p-2"><Button variant="ghost" size="icon" aria-label="Предыдущая страница" disabled={current === 0} onClick={() => setPage(current - 1)}><ChevronLeft size={14} /></Button><span className="text-xs text-muted">{current + 1} / {pages}</span><Button variant="ghost" size="icon" aria-label="Следующая страница" disabled={current === pages - 1} onClick={() => setPage(current + 1)}><ChevronRight size={14} /></Button></div>}
    </aside>
  );
}
