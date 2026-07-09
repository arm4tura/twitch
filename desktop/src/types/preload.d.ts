/**
 * Типы preload API (см. electron/preload.ts).
 *
 * Дублируем сигнатуру руками — Electron `contextBridge` не генерирует .d.ts,
 * а импортировать preload.ts в renderer нельзя (у него другой tsconfig и
 * рантайм). Ключевое правило: если добавляешь метод в preload.ts —
 * добавляй его и сюда.
 */

export {};

declare global {
  interface Window {
    twitchCut: {
      getBackendPort(): Promise<number>;
      getPathForFile(file: File): string;
      getGpuMode(): Promise<boolean>;
      onBootstrapProgress(cb: (p: unknown) => void): () => void;
      openBootstrapLog(): Promise<boolean>;
      retryBootstrap(): Promise<boolean>;
      openFile(opts?: {
        filters?: Array<{ name: string; extensions: string[] }>;
        title?: string;
      }): Promise<string | null>;
      openDirectory(): Promise<string | null>;
      saveFile(opts?: {
        defaultPath?: string;
        filters?: Array<{ name: string; extensions: string[] }>;
      }): Promise<string | null>;
      showInFolder(targetPath: string): Promise<boolean>;
      openPath(targetPath: string): Promise<boolean>;
      openExternal(url: string): Promise<boolean>;
    };
  }
}
