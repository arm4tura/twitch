import { useState, type ReactNode } from "react";
import { Folder, FolderOpen, Save } from "lucide-react";
import { Input } from "./Input";
import { Button } from "./Button";

type Kind = "file" | "directory" | "save";
type Filter = { name: string; extensions: string[] };

interface FileFieldProps {
  value: string;
  onChange: (v: string) => void;
  kind: Kind;
  filters?: Filter[];
  placeholder?: string;
  defaultPath?: string;
  invalid?: boolean;
  disabled?: boolean;
  /** Иконка слева от input'а. По умолчанию — папка/файл в зависимости от kind. */
  leftIcon?: ReactNode;
}

/**
 * FileField — единая обёртка над native-dialog'ами preload'а. Заменяет
 * fragmented `FileRow` из старого ImportScreen'а. Один компонент → одна
 * визуальная норма для всех выборов пути.
 */
export function FileField({
  value,
  onChange,
  kind,
  filters,
  placeholder,
  defaultPath,
  invalid,
  disabled,
  leftIcon,
}: FileFieldProps) {
  const [busy, setBusy] = useState(false);

  const pick = async () => {
    setBusy(true);
    try {
      let picked: string | null = null;
      if (kind === "file") {
        picked = await window.twitchCut.openFile({ filters });
      } else if (kind === "directory") {
        picked = await window.twitchCut.openDirectory();
      } else {
        picked = await window.twitchCut.saveFile({
          defaultPath: defaultPath ?? (value || undefined),
          filters,
        });
      }
      if (picked) onChange(picked);
    } finally {
      setBusy(false);
    }
  };

  const icon =
    leftIcon ??
    (kind === "directory" ? (
      <Folder className="h-4 w-4" />
    ) : kind === "save" ? (
      <Save className="h-4 w-4" />
    ) : (
      <FolderOpen className="h-4 w-4" />
    ));

  return (
    <div className="flex items-stretch gap-2">
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        leftIcon={icon}
        invalid={invalid}
        disabled={disabled}
        spellCheck={false}
      />
      <Button
        type="button"
        variant="secondary"
        size="md"
        onClick={pick}
        loading={busy}
        disabled={disabled}
      >
        Выбрать…
      </Button>
    </div>
  );
}
