import type { PropsWithChildren, ReactNode } from 'react';

export function StepLayout({
  title,
  subtitle,
  completeButton,
  backButton,
  children,
}: PropsWithChildren<{
  backButton?: ReactNode;
  completeButton?: ReactNode;
  subtitle: string;
  title: string;
}>) {
  return (
    <main className="flex flex-col">
      <div className="flex-1 p-6 pb-0">
        <div className="mx-auto max-w-2xl space-y-6">
          <div>
            <h3 className="text-2xl font-semibold">{title}</h3>
            <p className="mt-2 text-muted-foreground">{subtitle}</p>
          </div>

          <div className="space-y-4">{children}</div>
        </div>
      </div>

      <div className="p-6">
        <div className="mx-auto flex max-w-2xl justify-between">
          {backButton}
          <div className="grow" />
          {completeButton}
        </div>
      </div>
    </main>
  );
}
