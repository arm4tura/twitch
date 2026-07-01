/**
 * Preload: единственный мост между renderer (браузер) и main (Node.js).
 *
 * contextIsolation=true — из renderer нельзя дёрнуть Node API напрямую.
 * Всё, что должен уметь фронт, вешаем на `window.twitchCut` через
 * contextBridge.exposeInMainWorld. Типы этого API дублируются в
 * src/types/preload.d.ts, чтобы React мог их видеть.
 */

import { contextBridge, ipcRenderer } from "electron";

const api = {
  /** Порт FastAPI backend'а (выбран uvicorn'ом при старте). */
  getBackendPort(): Promise<number> {
    return ipcRenderer.invoke("get-backend-port");
  },

  /** Native OS dialog: выбрать файл. */
  openFile(opts?: {
    filters?: Array<{ name: string; extensions: string[] }>;
    title?: string;
  }): Promise<string | null> {
    return ipcRenderer.invoke("dialog:openFile", opts);
  },

  /** Native OS dialog: выбрать каталог. */
  openDirectory(): Promise<string | null> {
    return ipcRenderer.invoke("dialog:openDirectory");
  },

  /** Native OS dialog: сохранить как. */
  saveFile(opts?: {
    defaultPath?: string;
    filters?: Array<{ name: string; extensions: string[] }>;
  }): Promise<string | null> {
    return ipcRenderer.invoke("dialog:saveFile", opts);
  },

  /**
   * Открыть каталог с выделением файла в системном проводнике
   * (Explorer / Finder / Nautilus). Возвращает true, если операция удалась.
   * Тихая ошибка если путь не существует — main логирует, renderer
   * показывает toast «не удалось открыть».
   */
  showInFolder(targetPath: string): Promise<boolean> {
    return ipcRenderer.invoke("shell:showInFolder", targetPath);
  },

  /**
   * Открыть КАТАЛОГ (или файл) как есть — Explorer/Finder покажет содержимое
   * папки, а не подсветит её родителем. Для «Открыть папку логов».
   */
  openPath(targetPath: string): Promise<boolean> {
    return ipcRenderer.invoke("shell:openPath", targetPath);
  },

  /**
   * Открыть внешнюю ссылку (http/https/mailto) в системном браузере.
   * Main фильтрует по протоколу — `file://` через это API не пройдёт,
   * чтобы никто не смог из renderer'а стрельнуть в произвольный файл.
   */
  openExternal(url: string): Promise<boolean> {
    return ipcRenderer.invoke("shell:openExternal", url);
  },
};

contextBridge.exposeInMainWorld("twitchCut", api);
