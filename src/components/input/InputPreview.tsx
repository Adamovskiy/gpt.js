export function InputPreview({ fileContent, fileName }: { fileContent: string; fileName: string }) {
  return (
    <>
      <div className="text-sm font-medium">Current file: {fileName}</div>
      <div className="bg-muted p-3 rounded-md">
        <div className="text-xs text-muted-foreground mb-1">Preview (first 200 characters):</div>
        <pre className="text-sm whitespace-pre-wrap overflow-hidden">{fileContent.slice(0, 200)}...</pre>
      </div>
    </>
  );
}
