// Port of core/chunking.split_text — identical targets so inline-KB chunks
// look like Studio-built chunks.
const TARGET = 1200;
const OVERLAP = 150;

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+/).filter(s => s.trim());
}

export function splitText(raw: string, target = TARGET, overlap = OVERLAP): string[] {
  const text = raw.replace(/\r\n?/g, '\n');
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
