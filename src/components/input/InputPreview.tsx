export function InputPreview({ fileContent, fileName }: { fileContent: string; fileName: string }) {
  return (
    <>
      <div className="text-sm font-medium">Current file: {fileName}</div>
      <div className="text-sm font-medium">File size: {fileContent.length} chars</div>
      <div className="rounded-md bg-muted p-3">
        <div className="mb-1 text-xs text-muted-foreground">Preview (first 200 characters):</div>
        <pre className="overflow-hidden text-sm whitespace-pre-wrap">{fileContent.slice(0, 200)}...</pre>
      </div>
    </>
  );
}
