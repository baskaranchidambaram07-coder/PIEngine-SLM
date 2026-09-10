// Port of core/chunking.split_text — identical targets and identical section
// handling, so inline-KB chunks look like Studio-built chunks. Change both
// files together or a document uploaded on the phone will chunk differently
// from the same document uploaded in the Studio.
const TARGET = 1200;
const OVERLAP = 150;

// A heading line: markdown (## Foo), a numbered section (3. Foo, 2.1. Foo), or
// a short ALL-CAPS line. The length cap and trailing-punctuation test keep
// numbered LIST items ("1. Confirm the UAT plan for August.") from being read
// as headings — those are common in meeting minutes and splitting on them
// would shred the document.
const MAX_HEADING_CHARS = 80;

const HEADING = new RegExp(
  '^(?:' +
  '#{1,6}\\s+\\S.*' +                 // markdown heading
  '|\\d+(?:\\.\\d+)*\\.\\s+[^\\d\\s].*' + // 1. Section  /  2.3. Section
  "|[A-Z][A-Z0-9 \\-/&']{3,}" +       // SHORT ALL-CAPS HEADING
  ')$',
);

export function isHeading(raw: string): boolean {
  const line = raw.trim();
  if (!line || line.length > MAX_HEADING_CHARS) return false;
  if ('.,;'.includes(line[line.length - 1])) return false;  // a sentence, not a title
  return HEADING.test(line);
}

/** [heading, body] pairs; heading is '' for text before the first one. */
export function splitSections(text: string): Array<[string, string]> {
  const sections: Array<[string, string[]]> = [['', []]];
  for (const line of text.split('\n')) {
    if (isHeading(line)) sections.push([line.trim(), []]);
    else sections[sections.length - 1][1].push(line);
  }
  return sections
    .map(([h, b]) => [h, b.join('\n').trim()] as [string, string])
    .filter(([, b]) => b.length > 0);
}

/**
 * Split into chunks that never straddle a section heading.
 *
 * Merging purely on character count put a document's section 8 (a table of
 * engineering machines) and section 9 (approval gates) into ONE chunk, and the
 * model then answered a question about gate G2 with a sentence about the
 * Change machine. Sections are hard boundaries, and every chunk carries its
 * heading so both the retriever and the model can see where the text is from.
 */
export function splitText(raw: string, target = TARGET, overlap = OVERLAP): string[] {
  const text = raw.replace(/\r\n?/g, '\n');
  const out: string[] = [];
  for (const [heading, body] of splitSections(text)) {
    const prefix = heading ? `${heading}\n\n` : '';
    const budget = Math.max(200, target - prefix.length);
    for (const chunk of splitBlock(body, budget, overlap)) out.push(prefix + chunk);
  }
  return out;
}

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+/).filter(s => s.trim());
}

/** Paragraph-first recursive splitter with sentence fallback and overlap. */
function splitBlock(text: string, target: number, overlap: number): string[] {
  const paragraphs = text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);

  const pieces: string[] = [];
  for (const para of paragraphs) {
    if (para.length <= target) {
      pieces.push(para);
      continue;
    }
    let buf = '';
    for (const sent of splitSentences(para)) {
      if (buf && buf.length + sent.length + 1 > target) {
        pieces.push(buf);
        buf = overlap ? buf.slice(-overlap) + ' ' + sent : sent;
      } else {
        buf = (buf + ' ' + sent).trim();
      }
    }
    if (buf) pieces.push(buf);
  }

  const chunks: string[] = [];
  let buf = '';
  for (const piece of pieces) {
    if (buf && buf.length + piece.length + 2 > target) {
      chunks.push(buf);
      buf = piece;
    } else {
      buf = (buf + '\n\n' + piece).trim();
    }
  }
  if (buf) chunks.push(buf);
  return chunks;
}
