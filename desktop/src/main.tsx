import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { TooltipProvider } from "./components/ui/Tooltip";
import "./styles.css";

/**
 * TooltipProvider на root — Radix требует один провайдер выше по дереву,
 * иначе все <Tooltip> кричат в консоль. delayDuration=200 — быстрее дефолта,
 * ощущается отзывчивее.
 *
 * <Toaster> монтируется РОВНО ОДИН раз — внутри <App> (App.tsx). sonner рисует
 * каждый toast() в каждом смонтированном Toaster'е, поэтому второй экземпляр
 * здесь давал дубль всех уведомлений («сохранено» ×2). Держим один источник.
 */
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <TooltipProvider delayDuration={200}>
      <App />
    </TooltipProvider>
  </React.StrictMode>
);
