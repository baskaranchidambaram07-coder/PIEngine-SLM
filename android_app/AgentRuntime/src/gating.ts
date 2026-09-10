// Decide whether a message should hit the knowledge base at all.
//
// Line-for-line port of core/gating.py — change both together or the device
// and the server will answer the same message differently.
//
// Retrieval used to run unconditionally, so a bare "hi" was embedded, searched,
// and answered with four chunks of context (~1000 prompt tokens). The obvious
// patch — raise rag.min_score — does not generalise: cosine from a bi-encoder
// is a RANKING function, not a relevance detector. Its embeddings are
// anisotropic, so unrelated text still scores ~0.45-0.55 and there is no "no
// match" value to threshold against. Measured on this project's own two KBs,
// "thanks!" (0.5553) outranks the real question "what did we decide about the
// architecture?" (0.5547) — no threshold separates them.
//
// So this gate is deterministic and lexical. It answers only what it can answer
// safely: is there anything here to look up? It is intentionally conservative
// and must never suppress a real question. Off-topic *questions* ("what is the
// weather in Chennai?") are NOT caught here — no lexical rule catches those.

const STOPWORDS = new Set(
  ('a an the is are was were be been being am do does did doing have has had having ' +
   'what which who whom whose when where why how this that these those i you he she ' +
   'it we they me him her us them my your his her its our their to of in on at for ' +
   'with about from by as and or but if then than so not no nor too very can could ' +
   'will would shall should may might must there here all any some more most other ' +
   'into over under again further once').split(' '),
);

// A message made only of these is social, not a query. Deliberately short:
// every addition is a chance to swallow a real question.
const GREETING_WORDS = new Set(
  ('hi hello hey heya yo hiya greetings morning afternoon evening night good ' +
   'thanks thank thankyou thx ty cheers appreciated welcome ' +
   'ok okay okey k sure yeah yep yes nope nah fine cool nice great awesome perfect ' +
   'bye goodbye later farewell see ya cya ' +
   'please sorry apologies oops hmm hm ah oh wow').split(' '),
);

const MAX_GREETING_WORDS = 5;

/** Lowercase alphanumeric tokens with function words removed. */
export function contentWords(text: string): string[] {
  const words = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  return words.filter(w => w.length > 2 && !STOPWORDS.has(w));
}

/** [retrieve?, reason] — reason goes to the in-app diagnostic log. */
export function shouldRetrieve(query: string): [boolean, string] {
  if (!query || !query.trim()) return [false, 'empty message'];

  const words = contentWords(query);
  if (!words.length) return [false, 'no content words'];

  if (words.length <= MAX_GREETING_WORDS && words.every(w => GREETING_WORDS.has(w))) {
    return [false, 'greeting only'];
  }

  return [true, 'has content'];
}
