// SAM3 reference cutout client: queues a tiny 1-node `nynxz.Layout.Cutout` job through ComfyUI's
// execution queue (just like `agent.ts`) so SAM3 loads safely under dynamic VRAM, then returns the
// saved cutout's `input` ref and dimensions.

import { api } from "@comfy/api";

import { route } from "@/framework";
import type { FrescoPoint } from "@/lib/frescoScene";
import type { SourceType } from "@/lib/mediaRefs";

export type CutoutBackground = "white" | "grey" | "black" | "transparent";

export interface CutoutRequest {
  imageRef: string;
  imageType: SourceType;
  sam3: string;
  prompt: string;
  points: FrescoPoint[];
  background: CutoutBackground;
  crop: boolean;
  grow: number;
  feather: number;
}

export interface CutoutResult {
  ref: string;
  type: SourceType;
  width: number;
  height: number;
}

interface ComfyApi {
  clientId?: string;
  fetchApi(path: string, init?: RequestInit): Promise<Response>;
  addEventListener(type: string, fn: (e: Event) => void): void;
  removeEventListener(type: string, fn: (e: Event) => void): void;
}
const comfy = api as unknown as ComfyApi;

const STORAGE_KEY = "nynxz.cutout.sam3";

export function savedSam3Checkpoint(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function rememberSam3Checkpoint(name: string): void {
  try {
    if (name) localStorage.setItem(STORAGE_KEY, name);
  } catch {
    // ignore storage errors in private contexts
  }
}

export async function listSam3Checkpoints(): Promise<string[]> {
  try {
    const res = await fetch(route("layouts/sam3-checkpoints"));
    const body = (await res.json().catch(() => ({}))) as { checkpoints?: string[] };
    return Array.isArray(body.checkpoints) ? body.checkpoints : [];
  } catch {
    return [];
  }
}

/** Run `nynxz.Layout.Cutout` on ComfyUI's queue and resolve with the saved cutout ref. */
export function runCutout(req: CutoutRequest, signal?: AbortSignal): Promise<CutoutResult> {
  const prompt = {
    cutout: {
      class_type: "nynxz.Layout.Cutout",
      inputs: {
        image_ref: req.imageRef,
        image_type: req.imageType,
        sam3: req.sam3,
        prompt: req.prompt,
        points: JSON.stringify(req.points),
        background: req.background,
        crop: req.crop,
        grow: Math.round(req.grow),
        feather: Math.max(0, Math.round(req.feather)),
      },
    },
  };

  return new Promise((resolve, reject) => {
    let promptId = "";
    const cleanup = () => {
      comfy.removeEventListener("executed", onExecuted);
      comfy.removeEventListener("execution_error", onError);
      comfy.removeEventListener("execution_interrupted", onStopped);
      comfy.removeEventListener("execution_success", onSuccess);
      signal?.removeEventListener("abort", onAbort);
    };

    const extract = (output?: { cutout?: CutoutResult[] }): CutoutResult | null => {
      const first = output?.cutout?.[0];
      return first?.ref ? first : null;
    };

    const onExecuted = (e: Event) => {
      const d = (
        e as CustomEvent<{
          prompt_id?: string;
          node?: string;
          output?: { cutout?: CutoutResult[] };
        }>
      ).detail;
      if (!promptId || d?.prompt_id !== promptId || d.node !== "cutout") return;
      const hit = extract(d.output);
      cleanup();
      if (hit) resolve(hit);
      else reject(new Error("Cutout finished without returning an image."));
    };

    const onSuccess = (e: Event) => {
      const d = (e as CustomEvent<{ prompt_id?: string }>).detail;
      if (!promptId || d?.prompt_id !== promptId) return;
      // Fallback for cached executions where `executed` doesn't re-fire.
      void comfy
        .fetchApi(`/history/${encodeURIComponent(promptId)}`)
        .then((r) => r.json())
        .then((hist: Record<string, { outputs?: { cutout?: { cutout?: CutoutResult[] } } }>) => {
          const hit = extract(hist?.[promptId]?.outputs?.cutout);
          cleanup();
          if (hit) resolve(hit);
          else reject(new Error("Cutout returned no result."));
        })
        .catch(() => {
          cleanup();
          reject(new Error("Cutout finished without returning an image."));
        });
    };

    const onError = (e: Event) => {
      const d = (e as CustomEvent<{ prompt_id?: string; exception_message?: string }>).detail;
      if (!promptId || d?.prompt_id !== promptId) return;
      cleanup();
      reject(new Error(d.exception_message?.trim() || "SAM3 cutout failed."));
    };

    const onStopped = (e: Event) => {
      if (!promptId || (e as CustomEvent<{ prompt_id?: string }>).detail?.prompt_id !== promptId)
        return;
      cleanup();
      reject(new Error("Stopped."));
    };

    const onAbort = () => {
      cleanup();
      if (promptId) {
        void comfy.fetchApi("/queue", {
          method: "POST",
          body: JSON.stringify({ delete: [promptId] }),
        });
        void comfy.fetchApi("/interrupt", {
          method: "POST",
          body: JSON.stringify({ prompt_id: promptId }),
        });
      }
      reject(new Error("Stopped."));
    };

    comfy.addEventListener("executed", onExecuted);
    comfy.addEventListener("execution_error", onError);
    comfy.addEventListener("execution_interrupted", onStopped);
    comfy.addEventListener("execution_success", onSuccess);
    signal?.addEventListener("abort", onAbort);

    void comfy
      .fetchApi("/prompt", {
        method: "POST",
        body: JSON.stringify({ prompt, client_id: comfy.clientId }),
      })
      .then(async (res) => {
        const body = (await res.json()) as {
          prompt_id?: string;
          error?: { message?: string };
          node_errors?: Record<string, { errors?: { message: string; details?: string }[] }>;
        };
        if (!res.ok || !body.prompt_id) {
          const nodeError = Object.values(body.node_errors ?? {})[0]?.errors?.[0];
          cleanup();
          reject(
            new Error(
              nodeError
                ? `${nodeError.message}${nodeError.details ? ` (${nodeError.details})` : ""}`
                : (body.error?.message ?? "ComfyUI refused the cutout job."),
            ),
          );
          return;
        }
        promptId = body.prompt_id;
      })
      .catch((e: unknown) => {
        cleanup();
        reject(e instanceof Error ? e : new Error(String(e)));
      });
  });
}
