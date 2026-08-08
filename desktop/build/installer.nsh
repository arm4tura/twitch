; Кастомный макрос деинсталлятора Twitch Cut.
; Вызывается electron-builder (nsis.include) в начале секции удаления,
; до того как основной деинсталлятор трогает файлы программы.
;
; Спрашивает пользователя, удалять ли скачанные данные (~10 ГБ):
;   - $INSTDIR\TwitchCutter-Data  (portable-режим: данные рядом с .exe)
;   - $APPDATA\Twitch Cutter      (обычная установка: %APPDATA%)
;
; RMDir /r на несуществующем пути — no-op, ошибки не будет.

!macro customUnInstall
  MessageBox MB_YESNO|MB_ICONQUESTION \
    "Удалить все скачанные данные Twitch Cut (~10 ГБ)?$\n$\nБудут удалены Python, модели распознавания и виртуальное окружение.$\nСами файлы программы удаляются отдельно." \
    IDNO twitch_cut_wipe_skip
    RMDir /r "$INSTDIR\TwitchCutter-Data"
    RMDir /r "$APPDATA\Twitch Cutter"
  twitch_cut_wipe_skip:
!macroend
