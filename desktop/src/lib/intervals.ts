/** Union of intervals, without mutating source data or double-counting overlaps. */
export function mergeIntervals(segments: [number, number][]): [number, number][] {
  const result: [number, number][] = [];
  for (const [start, end] of segments.filter(([s, e]) => Number.isFinite(s) && Number.isFinite(e) && e > s).sort((a, b) => a[0] - b[0])) {
    const last = result[result.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else result.push([start, end]);
  }
  return result;
}
