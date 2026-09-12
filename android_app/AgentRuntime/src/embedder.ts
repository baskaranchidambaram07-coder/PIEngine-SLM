// Which embedding model an agent's knowledge base was built with.
//
// The Studio picks one per agent from core/embedders.py and ships its full
// descriptor in manifest.embedder: the GGUF to download, its dimension, the
// pooling and the query/passage prefixes. Embedding a question with anything
// else would search the KB with the wrong vector, so everything on the device
// that embeds (chat.ts, attachments.ts) goes through `embedderOf(manifest)`.
//
// Bundles published before the catalogue existed carry only {id, dim} or
// nothing; they were all built with bge-small, which is the default here.
import { EMBEDDER_FILE, EMBEDDER_URL, EMBEDDING_DIM, QUERY_PREFIX } from './config';

export type EmbedderSpec = {
  id: string;
  file: string;
  download_url: string;
  size_bytes?: number;
  dim: number;
  pooling: 'cls' | 'mean' | 'none' | 'last';
  query_prefix: string;
  passage_prefix: string;
};

export const DEFAULT_EMBEDDER: EmbedderSpec = {
  id: 'bge-small-en-v1.5',
  file: EMBEDDER_FILE,
  download_url: EMBEDDER_URL,
  size_bytes: 36806944,
  dim: EMBEDDING_DIM,
  pooling: 'cls',
  query_prefix: QUERY_PREFIX,
  passage_prefix: '',
};

export function embedderOf(manifest: any): EmbedderSpec {
  const e = manifest?.embedder;
  if (!e || !e.file) return DEFAULT_EMBEDDER;
  return {
    id: String(e.id || DEFAULT_EMBEDDER.id),
    file: String(e.file),
    download_url: String(e.download_url || ''),
    size_bytes: Number(e.size_bytes) || undefined,
    dim: Number(e.dim) || DEFAULT_EMBEDDER.dim,
    pooling: (e.pooling === 'mean' || e.pooling === 'cls' || e.pooling === 'last' || e.pooling === 'none') ? e.pooling : 'cls',
    query_prefix: typeof e.query_prefix === 'string' ? e.query_prefix : '',
    passage_prefix: typeof e.passage_prefix === 'string' ? e.passage_prefix : '',
  };
}
