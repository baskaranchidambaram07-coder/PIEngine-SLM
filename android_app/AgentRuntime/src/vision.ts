// Optional on-device vision model for attached images — the handset port of
// runtime/vision.py.
//
// OFF by default and gated behind a setting, because the model is another
// ~1.5 GB on top of the agent's own and it needs headroom the 6 GB test phone
// does not have (Qwen3-1.7B text-only already fails to init there). Images
// still work without it: ML Kit OCR reads the text and the agent's model
// answers from that. With it on, the vision model sees the pixels AND the
// OCR transcript, exactly as the web runtime does.
//
// The single-context rule of src/llm.ts applies: loading the vision model
// releases the chat model, and the next text turn reloads it. The same
// blocked-model safety gate protects the app from repeat-init crashes.
import AsyncStorage from '@react-native-async-storage/async-storage';
import ReactNativeBlobUtil from 'react-native-blob-util';
import { MODELS_DIR } from './config';
import { completionWithImage, CompletionStats, ensureVisionModel, StreamEvent } from './llm';
import { downloadWithFallback, getPortalUrl, Progress } from './portal';
import { log } from './logger';

const fs = ReactNativeBlobUtil.fs;
const KEY = 'vision_enabled_v1';

export const VISION_MODEL = {
  id: 'qwen3-vl-2b-q4_k_m',
  name: 'Qwen3-VL 2B (Q4_K_M)',
  file: 'Qwen3-VL-2B-Instruct-Q4_K_M.gguf',
  mmproj: 'mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf',
  file_url: 'https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3-VL-2B-Instruct-Q4_K_M.gguf',
  mmproj_url: 'https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf',
  size_gb: 1.55,
  min_device_ram_gb: 8,
};

export async function isVisionEnabled(): Promise<boolean> {
  return (await AsyncStorage.getItem(KEY)) === '1';
}

export async function setVisionEnabled(on: boolean): Promise<void> {
  await AsyncStorage.setItem(KEY, on ? '1' : '0');
  await log(`vision: ${on ? 'enabled' : 'disabled'} by user`);
}

export async function visionFilesReady(): Promise<boolean> {
  return (await fs.exists(`${MODELS_DIR}/${VISION_MODEL.file}`))
      && (await fs.exists(`${MODELS_DIR}/${VISION_MODEL.mmproj}`));
}

/** Vision is used only when the user turned it on AND both files are present. */
export async function visionAvailable(): Promise<boolean> {
  return (await isVisionEnabled()) && (await visionFilesReady());
}

/** Fetch both GGUF files — portal cache first (same files the web runtime
 *  uses), Hugging Face as fallback. Large-file path via Download Manager. */
export async function downloadVision(onProgress?: Progress): Promise<void> {
  const base = await getPortalUrl();
  await fs.mkdir(MODELS_DIR).catch(() => {});
  const jobs: Array<[string, string, string]> = [
    [VISION_MODEL.file, VISION_MODEL.file_url, `Vision model (${VISION_MODEL.size_gb} GB)`],
    [VISION_MODEL.mmproj, VISION_MODEL.mmproj_url, 'Vision projector (0.45 GB)'],
  ];
  for (const [file, hf, label] of jobs) {
    const dest = `${MODELS_DIR}/${file}`;
    if (await fs.exists(dest)) continue;
    await downloadWithFallback([`${base}/models/${encodeURIComponent(file)}`, hf], dest, label, true, onProgress);
  }
  await log('vision: files downloaded');
}

export async function removeVisionFiles(): Promise<void> {
  for (const f of [VISION_MODEL.file, VISION_MODEL.mmproj]) {
    await fs.unlink(`${MODELS_DIR}/${f}`).catch(() => {});
  }
}

/** Answer `question` about the image at `imagePath`, streaming tokens. */
export async function answerWithImage(
  imagePath: string, question: string, systemPrompt: string, ocrText: string,
  generation: any, onEvent: (ev: StreamEvent) => void, onStatus?: (s: string) => void,
): Promise<{ text: string; stats: CompletionStats | null }> {
  const ctx = await ensureVisionModel(VISION_MODEL.file, VISION_MODEL.mmproj, onStatus);
  const text = ocrText
    ? `${question}\n\n(OCR transcript of the same image, for reference — trust the image where they disagree:)\n${ocrText.slice(0, 3000)}`
    : question;
  onStatus?.('Reading the image with the vision model…');
  return completionWithImage(ctx, systemPrompt, text, imagePath,
    { ...(generation || {}), temperature: 0.3, max_tokens: 512 }, onEvent);
}
