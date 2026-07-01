import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Комбинирует условные классы (clsx) и мерджит конфликтующие Tailwind-утилиты
 * (tailwind-merge). Стандартный shadcn-паттерн: `cn("p-4", condition && "p-2")`
 * даст `p-2`, а не `p-4 p-2`.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
