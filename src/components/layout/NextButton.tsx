import { ChevronRight, Loader } from 'lucide-react';

import { Button } from '@/components/ui/button.tsx';

export function NextButton({
  onClick,
  title,
  disabled,
  loading,
}: {
  disabled?: boolean;
  loading?: boolean;
  onClick: () => void;
  title: string;
}) {
  return (
    <Button disabled={disabled} onClick={onClick}>
      {loading && <Loader className="mr-2 size-4 animate-spin" />}
      {title}
      {!loading && <ChevronRight className="ml-1 size-4" />}
    </Button>
  );
}
