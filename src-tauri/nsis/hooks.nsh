!macro NSIS_HOOK_POSTUNINSTALL
  ; Remove AuraSeek user data (DB/models/logs) for a "clean uninstall"
  RMDir /r "$APPDATA\auraseek"
  RMDir /r "$LOCALAPPDATA\auraseek"
!macroend

