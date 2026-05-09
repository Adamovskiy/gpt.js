import { useMemo, useState } from 'react';

import { Button } from '@/components/ui/button';

import type { CharTokenizer } from '@/tokenizer.ts';

const PAGE_SIZE = 36;

export function Vocabulary({ tokenizer }: { tokenizer: CharTokenizer }) {
  const [page, setPage] = useState(0);

  const totalItems = tokenizer.vocabulary.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / PAGE_SIZE));

  const currentPage = Math.min(page, totalPages - 1);

  const visibleVocabulary = useMemo(() => {
    const start = currentPage * PAGE_SIZE;
    return tokenizer.vocabulary.slice(start, start + PAGE_SIZE);
  }, [tokenizer.vocabulary, currentPage]);

  const from = totalItems === 0 ? 0 : currentPage * PAGE_SIZE + 1;
  const to = Math.min((currentPage + 1) * PAGE_SIZE, totalItems);

  return (
    <div className="w-full rounded-xl border bg-card p-3">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium">tokenizer.vocabulary</h3>
          <p className="text-xs text-muted-foreground">
            {from}–{to} of {totalItems} tokens
          </p>
        </div>

        {totalPages > 1 && (
          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => setPage((page) => Math.max(0, page - 1))}
              disabled={currentPage === 0}
            >
              Prev
            </Button>

            <span className="min-w-16 text-center text-xs text-muted-foreground">
              {currentPage + 1} / {totalPages}
            </span>

            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => setPage((page) => Math.min(totalPages - 1, page + 1))}
              disabled={currentPage >= totalPages - 1}
            >
              Next
            </Button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
        {visibleVocabulary.map((token, localIdx) => {
          const idx = currentPage * PAGE_SIZE + localIdx;
          const value = JSON.stringify(token).slice(1, -1);

          return (
            <div
              key={`${idx}-${value}`}
              className="grid grid-cols-[3rem_1fr] overflow-hidden rounded-md bg-gray-50 text-sm"
            >
              <div className="border-r bg-muted px-2 py-1.5 font-mono text-xs text-muted-foreground">{idx}</div>

              <div className="truncate px-2 py-1.5 font-mono" title={value}>
                {value}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
