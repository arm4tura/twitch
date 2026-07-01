import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { TooltipProvider } from "./components/ui/Tooltip";
import { Toaster } from "./components/ui/Toast";
import "./styles.css";

/**
 * TooltipProvider на root — Radix требует один провайдер выше по дереву,
 * иначе все <Tooltip> кричат в консоль. delayDuration=200 — быстрее дефолта,
 * ощущается отзывчивее.
 * Toaster рендерится как сосед — портал сам налетит на body.
 */
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <TooltipProvider delayDuration={200}>
      <App />
      <Toaster />
    </TooltipProvider>
  </React.StrictMode>
);
