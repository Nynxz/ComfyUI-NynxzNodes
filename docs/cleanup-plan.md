# Cleanup plan: fusion, layout, regions (and Fresco into Layout)

Status: **agreed, in progress**. Built from three read-only audits (2026-10-20). Mark up anything you
disagree with; nothing below is done yet except where it says so.

## Ground rules (every phase)

- **Saved workflows keep working.** Every registered node id keeps loading (`nynxz.Fusion.*`,
  `nynxz.Regions.*`, `nynxz.Fresco*`, `nynxz.Layout*`). Input **names and order** stay (widget values
  are stored by position too). Stored formats stay readable: Fresco scenes (dict or JSON string),
  Layout scenes per mode, `{}` = "choose a mode", library files, the `nynxzLayout` JSON envelope,
  the executed-UI keys (`fresco_*`, `layout_*`).
- **Outputs don't change** unless an item says so. Seed constants stay (e.g. fusion's `0x2545F491`).
- One concern per commit; checks + `pnpm check` green after each; nothing pushed without a go.

## Phase 0 — safety net first

1. **Golden fixtures** of real stored values: Fresco (dict and JSON string), Layout per mode,
   Layout with mixins, a `cast`-era value, a library file, the envelope. Python parse tests against
   them now, so later refactors can't silently change what old workflows load as.
2. **One shared test harness** (`scripts/_harness.py`): today there are five copy-pasted stub
   setups (two fusion, three regions/layout).
3. **Run the pure checks in CI** (`check_layout_*`, `check_fusion_alpha`, `check_fusion_qwen21`,
   `check_region_schedule`, `check_region_edit`). CI runs only `pnpm check` today, so none of the
   ~150 checks run unless by hand.

## Phase 1 — dead code and stale text (no behaviour change)

- **Fusion:** stale docstrings about an "original autogrow node" that never existed; "Fusion Weight
  Map"/"Fusion Studio"/"Fusion Regions Preview" references to things that don't exist; δ-Vision's last
  trace (`pyproject.toml` confusables); unused `sources` param, unused inspector payload keys (or show
  them), dead CSS var, wrong file paths in docstrings; package docstring (misses Fusion Regions).
- **Regions:** duplicate "no prompt and no LoRA" errors; `copy_full` param that's always True and the
  branches behind it; Z-Image's dead layout-lock check; dead `txtlen` store; unused `rank`/`energy`;
  undocumented, unread region `strength`; stale tooltips (LoKr "can't run", MOSS "prompts only",
  "region prompt replaces the main prompt"); profiling env var undocumented.
- **Layout/Fresco:** unused exports/consts (`BACKGROUND_ID`, …), unused `defineExpose` members,
  unused store options, dead CSS, "mixins's" typos, stale docstrings; NODES.md has no section for the
  Layout nodes and four region nodes (Grid, Split, Sketch, Overlay) are undocumented.
- **Done:** all of Phase 1 (54832a6…, ea03ae1, bb9e2dc and the NODES.md sections).

## Phase 2 — one copy of each shared thing

| What | Today | One home |
|---|---|---|
| Image listing API | `/nynxz/fusion/images` ≡ `/nynxz/images` | keep `/nynxz/images` (FusionGrid uses it) |
| Frontend media helpers | `fusionApi.ts` + `imagePickApi.ts` (types, cache, thumbs, upload) | `lib/mediaRefs.ts` |
| Loading reference images for an encode | Fresco `_references`, Layout `_encode_with_refs`, Extras `_open_ref` | one loader (tensor + PIL) |
| Refs/LoRAs parsing (Py) and readers (TS) | Fresco + Layout parsers, differing validation | one content parser per side |
| Qwen edit template | `fresco/_look.py` and `fusion/_encoders.py` | `_encoders` (goes with Krea edit, see Phase 5) |
| Mask → bounding box | 4 copies, `THRESHOLD=0.5` ×3 | `regions/_mask.py` |
| Pop-out host, popped-out summary, "send source image back" | Fresco + Layout copies | one composable/component; one Python helper |
| ids, colours, palettes, auto-naming, small utils | 2–3 copies each | shared util |
| Private names imported across packages | `_load_image`, `_gaussian_blur2d`, `_grown`, `_conditional_rows`, `_NEVER`… | public names in the right module |

## Phase 3 — regions engine

1. **Verify a likely Krea bug first:** Krea's `isolation` pushes *background* tokens apart; Z-Image
   and MOSS deliberately spare them (9215c36). Decide: fix Krea to match (output change on Krea with
   isolation > 0) or keep as is.
2. **Family registry** replacing the if/else dispatch and three near-copy `_apply_*` paths:
   `Family(name, detect, block_count, already_regioned, wanted, supports, install)`; generic
   "not on {family}" lines from `supports`.
3. **Shared base for the wrapper families** (Z-Image, MOSS): copies, layout, by-length gates,
   bias builders, caching, override chaining — today ~60% duplicated. MOSS keeps: canvas-first order,
   rope patch, split attention, `ffn_down` workaround. Z-Image keeps: padding, refiners, fused-qkv.
4. **One LoRA loader + one per-region cache** (Krea's and Z/MOSS's are two copies of the same steps).
5. **`apply_regions(model, regions, **settings)`** as a real function; Fresco stops calling the
   node's `execute` with hidden kwargs.
6. **Split the big modules** by responsibility: `apply.py` (node only), `_apply.py`, `_families.py`,
   Krea-only parts of `_patches.py`/`_state.py` into Krea modules, MOSS's `ffn_down` and profiling
   into their own.
7. **Tests:** a tiny Z-Image (NextDiT) CPU harness like the MOSS one (Z-Image has **no** tests today);
   Krea attention routing; `_mask`, `_lora` numerics against ComfyUI's merged weights.

## Phase 4 — fusion engine

- Split `_fusion.py` (1,792 lines): `_inputs.py` (widget builders — the only part needing
  `comfy_api`), `_weights.py` (pure weight field), `_images.py` (fit/alpha/grid); `_fusion.py` keeps
  encode orchestration, caches, prompt mode.
- `FusionSettings` built from the node's kwargs and the inspector payload from the settings (today
  every field is restated three times).
- Tests for what has none: Fusion Regions, regional pattern, "one copy" reading, the Qwen3.5/MOSS
  profiles, the inspector payload, jitter/shuffle/roll.

## Phase 5 — one editor (Fresco into Layout)

Keep (your answers): **painted shapes + SAM3**, **editing a source image** (denoise, background
denoise, layout start), **Overlay and Inputs**. Drop: **Krea edit / "read by look"** (`_look.py`, look
latents, `_refs` routing — once nothing else needs them; old "look" layers load as normal reference
reading, with a log note).

1. **One scene model** (TS + Python, one parser each): boxes with geometry, visibility, optional
   painted shape (+ grow, feather, SAM3 points/capture), words, kind/text/palette/tags, per-box refs,
   LoRAs, denoise; mixins; scene settings (caption, prompt format, style, MOSS meta, overlap,
   reference detail, prompt mode). Fixes on the way: `hidden` survives mode changes, `color` and
   `locked` survive parsing, `cast`-era values migrate.
2. **Readers for every old format** (Fresco scene → regions-format layout; Layout per mode; library
   files; envelope), tested against the Phase 0 fixtures.
3. **One editor host** on the existing canvas: the Layout editor gains painting/SAM3/drop-to-ref and
   per-box content; Fresco's editor and its duplicate components go.
4. **Nodes:** `nynxz.Layout` becomes the one editor node. Fresco's nodes stay registered (old
   workflows load and run) but are hidden from search and marked deprecated, ideally with a
   "convert to Layout" action. Decide how Fresco Apply's job is exposed for Layout (see decisions).

## Decisions (2026-10-20)

1. **Krea isolation:** fix it to spare the background like Z-Image and MOSS, after a test confirms
   the bug; the commit names the output change (Krea, isolation > 0).
2. **Fresco Apply's role:** a new **Layout Apply** node (model, positive, latent, regions, the edit
   path). Layout Extras stays the information node (prompt, overlay, sheet, regions for hand-wiring).
3. **Fresco nodes:** deprecated and hidden from search once Layout covers them; they keep loading.
4. **Order:** Phases 0–4 first, then Phase 5's model designed in detail.
