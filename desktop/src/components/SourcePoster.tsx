import { useState } from "react";
import { Film } from "lucide-react";
import { thumbnailUrl } from "../api";

export function SourcePoster({ path }: { path?: string | null }) {
  const [failed, setFailed] = useState(false);
  const src = path ? thumbnailUrl(path) : null;
  return <div className="source-poster">
    {src && !failed ? <img src={src} alt="Кадр исходной записи" onError={() => setFailed(true)} /> :
      <div><Film className="h-7 w-7" /><span>Превью недоступно</span></div>}
  </div>;
}
