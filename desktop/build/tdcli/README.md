# TwitchDownloaderCLI binaries (bundled for VOD download)

These folders hold the [TwitchDownloaderCLI](https://github.com/lay295/TwitchDownloader)
binary that the backend shells out to when downloading Twitch VODs
(`backend/src/twitch_cut/tdcli.py`). electron-builder ships the folder matching
the build's platform into `resources/tdcli/<platform>/` (see `package.json` →
`build.<os>.extraResources`), and `electron/main.ts` points the backend at it via
the `TWITCH_CUT_TDCLI_DIR` env var. Python then appends the per-platform subdir
itself (`get_tdcli_path`).

## Where to put what

Download the matching release asset from
https://github.com/lay295/TwitchDownloader/releases and drop the **binary**
(not the zip) into the folder for its platform, keeping the exact filename:

| Platform            | Folder                    | Binary name               |
| ------------------- | ------------------------- | ------------------------- |
| Windows x64         | `build/tdcli/win-x64/`    | `TwitchDownloaderCLI.exe` |
| macOS Apple Silicon | `build/tdcli/osx-arm64/`  | `TwitchDownloaderCLI`     |
| macOS Intel         | `build/tdcli/osx-x64/`    | `TwitchDownloaderCLI`     |
| Linux x64           | `build/tdcli/linux-x64/`  | `TwitchDownloaderCLI`     |

On unix the backend `chmod +x` the binary on first use, so you don't need to set
the executable bit yourself — but doing so doesn't hurt.

## Why these folders are committed empty

electron-builder fails the build if an `extraResources.from` directory is
missing. The `.gitkeep` files keep the folders present in a fresh checkout so
`npm run dist` doesn't error before you've fetched the binaries. The binaries
themselves are **not** committed (they're large and licensed separately) — see
`.gitignore`.

## macOS signing note

Under `hardenedRuntime: true` a nested unsigned executable in `Resources/` can be
killed by Gatekeeper once notarization is enabled. Current config has
`notarize: false`, so it runs as-is. When you turn notarization on, this binary
must be signed with the same certificate (electron-builder does **not** sign
extraResources automatically) or moved into the app's code-signing flow.

## Dev

In dev (`npm run dev`) the app is not packaged, so `resolveTdcliDir()` looks for
`backend/vendor/tdcli/<platform>/` instead. Drop a binary there to test VOD
download locally, or put `TwitchDownloaderCLI` on your `PATH` — the backend falls
back to `PATH` when no bundled binary is found.
