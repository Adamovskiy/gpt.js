import { useMemo, useState } from 'react';

import type { Tokenizer } from '@/llm/types.ts';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const PAGE_SIZE = 36;

export function Vocabulary({ tokenizer }: { tokenizer: Tokenizer }) {
  const [page, setPage] = useState(0);

  const totalItems = tokenizer.getVocabSize();
  const totalPages = Math.max(1, Math.ceil(totalItems / PAGE_SIZE));

  const currentPage = Math.min(page, totalPages - 1);

  const visibleVocabulary = useMemo(() => {
    const start = currentPage * PAGE_SIZE;
    return tokenizer.getVocab().slice(start, start + PAGE_SIZE);
  }, [tokenizer, currentPage]);

  const from = totalItems === 0 ? 0 : currentPage * PAGE_SIZE + 1;
  const to = Math.min((currentPage + 1) * PAGE_SIZE, totalItems);

  return (
    <Card>
      <CardHeader>
        <CardTitle>
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
                  disabled={currentPage === 0}
                  onClick={() => {
                    setPage((page) => Math.max(0, page - 1));
                  }}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  Prev
                </Button>

                <span className="min-w-16 text-center text-xs text-muted-foreground">
                  {currentPage + 1} / {totalPages}
                </span>

                <Button
                  disabled={currentPage >= totalPages - 1}
                  onClick={() => {
                    setPage((page) => Math.min(totalPages - 1, page + 1));
                  }}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  Next
                </Button>
              </div>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className="
          grid grid-cols-1 gap-2
          sm:grid-cols-2
          md:grid-cols-3
          lg:grid-cols-4
          xl:grid-cols-6
        "
        >
          {visibleVocabulary.map((token, localIdx) => {
            const idx = currentPage * PAGE_SIZE + localIdx;
            const value = JSON.stringify(token).slice(1, -1);

            return (
              <div
                className="
                grid grid-cols-[3rem_1fr] overflow-hidden rounded-md bg-gray-50
                text-sm
              "
                key={`${idx}-${value}`}
              >
                <div
                  className="
                  border-r bg-muted px-2 py-1.5 font-mono text-xs
                  text-muted-foreground
                "
                >
                  {idx}
                </div>

                <div className="truncate px-2 py-1.5 font-mono" title={value}>
                  {value}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
