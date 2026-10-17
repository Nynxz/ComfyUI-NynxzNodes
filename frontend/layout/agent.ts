// The layout agent: ask the vision-language model wired into Layout Inputs for a layout. Like H3
// Studio's rewriter, the job is queued like any run — the nodes feeding Layout Inputs' `clip`, the
// trace image if it's used, core's Generate Text and Preview as Text — so the model loads on the
// queue, never off it. The ask is built and the answer read by the backend
// (nodes/layout/_agent.py), through /nynxz/layouts/agent-*.

import { api } from "@comfy/api";
import { app } from "@comfy/app";

import { route } from "@/framework";
import type { SourceType } from "@/lib/fusionApi";

import type { PromptScene } from "./promptScene";

type ApiLink = [string, number];
interface ApiNode {
  class_type: string;
  inputs: Record<string, unknown>;
}
interface ComfyApi {
  clientId?: string;
  fetchApi(path: string, init?: RequestInit): Promise<Response>;
  addEventListener(type: string, fn: (e: Event) => void): void;
  removeEventListener(type: string, fn: (e: Event) => void): void;
}
const comfy = api as unknown as ComfyApi;

const LAYOUT = "nynxz.Layout";
const INPUTS = "nynxz.Layout.Inputs";
const IMAGE_SIDE = 768;

export interface AgentImage {
  ref: string;
  type: SourceType;
}
export interface AgentAsk {
  scene: PromptScene;
  /** The Layout node asking, so the model is the one wired into its own Layout Inputs. */
  nodeId: string;
  request: string;
  image: AgentImage | null;
  /** Start from the boxes there are now. */
  revise: boolean;
  seed: number;
  temperature: number;
}
export interface AgentLayout {
  prompt: string;
  background: string;
  boxes: {
    x: number;
    y: number;
    w: number;
    h: number;
    kind: "object" | "character" | "text";
    name: string;
    desc: string;
    text: string;
  }[];
}

const isLink = (v: unknown): v is ApiLink =>
  Array.isArray(v) && v.length === 2 && typeof v[1] === "number";

/** The nodes that make the CLIP wired into this Layout node's Layout Inputs, and its link — or
 *  null when there is none. */
export async function agentClip(
  nodeId: string,
): Promise<{ nodes: Record<string, ApiNode>; clip: ApiLink } | null> {
  const { output } = await (
    app as unknown as { graphToPrompt(): Promise<{ output: Record<string, ApiNode> }> }
  ).graphToPrompt();
  const key = Object.keys(output).find(
    (k) => output[k]!.class_type === LAYOUT && (k === nodeId || k.endsWith(`:${nodeId}`)),
  );
  const wired = key ? output[key]!.inputs.inputs : undefined;
  const inputs = isLink(wired) ? output[String(wired[0])] : undefined;
  const clip = inputs?.class_type === INPUTS ? inputs.inputs.clip : undefined;
  if (!isLink(clip)) return null;
  const nodes: Record<string, ApiNode> = {};
  const stack = [String(clip[0])];
  while (stack.length) {
    const id = stack.pop()!;
    if (id in nodes || !output[id]) continue;
    nodes[id] = output[id]!;
    for (const v of Object.values(output[id]!.inputs)) if (isLink(v)) stack.push(String(v[0]));
  }
  return { nodes, clip: [String(clip[0]), clip[1]] };
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(route(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const out = (await res.json().catch(() => ({}))) as T & { error?: string };
  if (!res.ok) throw new Error(out.error ?? `HTTP ${res.status}`);
  return out;
}

/** Queue `prompt`, whose answer is node `show` (Preview as Text), and wait for its text. */
function queueText(prompt: Record<string, ApiNode>, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    let id = "";
    const done = () => {
      comfy.removeEventListener("executed", onExecuted);
      comfy.removeEventListener("execution_error", onError);
      comfy.removeEventListener("execution_interrupted", onStopped);
      signal?.removeEventListener("abort", onAbort);
    };
    const onExecuted = (e: Event) => {
      const d = (
        e as CustomEvent<{ prompt_id: string; node: string; output?: { text?: string[] } }>
      ).detail;
      if (!id || d?.prompt_id !== id || d.node !== "show") return;
      done();
      resolve(d.output?.text?.[0] ?? "");
    };
    const onError = (e: Event) => {
      const d = (e as CustomEvent<{ prompt_id: string; exception_message?: string }>).detail;
      if (!id || d?.prompt_id !== id) return;
      done();
      reject(new Error(d.exception_message?.trim() || "The agent's job failed."));
    };
    const onStopped = (e: Event) => {
      if (!id || (e as CustomEvent<{ prompt_id: string }>).detail?.prompt_id !== id) return;
      done();
      reject(new Error("Stopped."));
    };
    const onAbort = () => {
      done();
      if (id) {
        void comfy.fetchApi("/queue", {
          method: "POST",
          body: JSON.stringify({ delete: [id] }),
        });
        void comfy.fetchApi("/interrupt", {
          method: "POST",
          body: JSON.stringify({ prompt_id: id }),
        });
      }
      reject(new Error("Stopped."));
    };
    comfy.addEventListener("executed", onExecuted);
    comfy.addEventListener("execution_error", onError);
    comfy.addEventListener("execution_interrupted", onStopped);
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
          done();
          reject(
            new Error(
              nodeError
                ? `${nodeError.message}${nodeError.details ? ` (${nodeError.details})` : ""}`
                : (body.error?.message ?? "ComfyUI refused the agent's job."),
            ),
          );
          return;
        }
        id = body.prompt_id;
      })
      .catch((e: unknown) => {
        done();
        reject(e instanceof Error ? e : new Error(String(e)));
      });
  });
}

/** Ask the agent for a layout. */
export async function askAgent(ask: AgentAsk, signal?: AbortSignal): Promise<AgentLayout> {
  const chain = await agentClip(ask.nodeId);
  if (!chain) {
    throw new Error(
      "Wire Layout Inputs into this node, with a vision-language CLIP (e.g. Qwen3-VL-8B from " +
        "Load CLIP) in its clip input.",
    );
  }
  const { text } = await post<{ text: string }>("layouts/agent-prompt", {
    layout: ask.scene,
    request: ask.request,
    revise: ask.revise,
    has_image: !!ask.image,
  });
  const nodes: Record<string, ApiNode> = { ...chain.nodes };
  if (ask.image) {
    nodes.picture = {
      class_type: "LoadImage",
      inputs: { image: `${ask.image.ref} [${ask.image.type}]` },
    };
    nodes.fit = {
      class_type: "ImageScaleToMaxDimension",
      inputs: { image: ["picture", 0], upscale_method: "area", largest_size: IMAGE_SIDE },
    };
  }
  nodes.write = {
    class_type: "TextGenerate",
    inputs: {
      clip: chain.clip,
      ...(ask.image ? { image: ["fit", 0] } : {}),
      prompt: text,
      max_length: 2048,
      sampling_mode: "on",
      "sampling_mode.temperature": ask.temperature,
      "sampling_mode.top_k": 64,
      "sampling_mode.top_p": 0.95,
      "sampling_mode.min_p": 0.05,
      "sampling_mode.repetition_penalty": 1.05,
      "sampling_mode.seed": ask.seed,
      thinking: false,
      use_default_template: true,
    },
  };
  nodes.show = { class_type: "PreviewAny", inputs: { source: ["write", 0] } };
  const reply = await queueText(nodes, signal);
  try {
    return await post<AgentLayout>("layouts/agent-read", { text: reply });
  } catch {
    throw new Error(`The answer had no boxes in it — try again. It said: ${reply.slice(0, 300)}`);
  }
}
