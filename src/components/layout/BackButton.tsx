import { ChevronLeft } from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';

export function BackButton({ onClick, disabled }: { disabled?: boolean; onClick: () => void }) {
  return (
    <Button disabled={disabled} onClick={onClick} variant="outline">
      <ChevronLeft className="mr-1 size-4" />
      Back
    </Button>
  );
}
