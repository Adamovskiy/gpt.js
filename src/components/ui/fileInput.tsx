import { Input } from './input.tsx';
import { Label } from './label.tsx';

export function FileInput({
  accept,
  label,
  onChange,
  disabled,
}: {
  accept: string;
  disabled?: boolean;
  label: string;
  onChange: (file: File | undefined) => void;
}) {
  return (
    <div
      className="
        space-y-2 rounded-lg border-2 border-dashed border-muted-foreground/25
        p-4
      "
    >
      <Label className="text-sm font-medium" htmlFor="file-upload">
        {label}
      </Label>
      <Input
        accept={accept}
        className="cursor-pointer"
        disabled={disabled}
        id="file-upload"
        onChange={(event) => {
          const file = event.target.files?.[0];
          onChange(file);
        }}
        type="file"
      />
    </div>
  );
}
