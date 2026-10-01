<template>
  <ZenWidget :gap="7" pad="6px 4px 4px" class="si">
    <!-- ── destination ───────────────────────────────────────────── -->
    <div class="si-field">
      <label class="si-lbl"><i class="mdi mdi-folder-outline" /> Project</label>
      <div class="si-project">
        <template v-if="!creating">
          <ZenCombo
            class="si-grow"
            :model-value="cfg.project"
            :items="projectItems"
            :menu-width="280"
            placeholder="output root"
            empty-text="No projects yet"
            @update:model-value="set('project', String($event))"
            @open="ensureProjects"
          >
            <template #selected>
              <span class="si-sel">
                <i class="mdi" :class="cfg.project ? 'mdi-folder' : 'mdi-folder-home-outline'" />
                <span class="si-sel-name">{{ cfg.project || "output root" }}</span>
              </span>
            </template>
            <template #option="{ item }">
              <i
                class="mdi si-opt-ic"
                :class="item.value ? 'mdi-folder' : 'mdi-folder-home-outline'"
              />
              <span class="si-opt-name">{{ item.label }}</span>
            </template>
            <template #footer="{ close }">
              <ZenButton
                class="si-newbtn"
                variant="ghost"
                sm
                block
                icon="mdi mdi-folder-plus-outline"
                @click="startCreate(close)"
                >New project…</ZenButton
              >
            </template>
          </ZenCombo>
          <ZenIconButton
            icon="mdi mdi-folder-plus-outline"
            title="New project"
            @click="startCreate()"
          />
        </template>
        <template v-else>
          <ZenInput
            ref="newInput"
            class="si-grow"
            :model-value="newName"
            placeholder="New project name…"
            sm
            @update:model-value="newName = String($event)"
            @keydown.enter="confirmCreate"
            @keydown.esc="cancelCreate"
          />
          <ZenIconButton icon="mdi mdi-check" title="Create" @click="confirmCreate" />
          <ZenIconButton icon="mdi mdi-close" title="Cancel" @click="cancelCreate" />
        </template>
      </div>
    </div>

    <div class="si-field">
      <label class="si-lbl"><i class="mdi mdi-source-branch" /> Subfolder</label>
      <ZenInput
        :model-value="cfg.subfolder"
        placeholder="optional, e.g. renders"
        sm
        @update:model-value="set('subfolder', String($event))"
      />
    </div>

    <div class="si-field">
      <label class="si-lbl"><i class="mdi mdi-tag-outline" /> Prefix</label>
      <ZenInput
        :model-value="cfg.filename_prefix"
        placeholder="optional, supports %date%…"
        sm
        @update:model-value="set('filename_prefix', String($event))"
      />
    </div>

    <div class="si-sep" />

    <!-- ── format + quality ──────────────────────────────────────── -->
    <div class="si-row">
      <label class="si-lbl"><i class="mdi mdi-file-image-outline" /> Format</label>
      <ZenToggleGroup
        :model-value="cfg.format"
        :options="FORMAT_OPTS"
        @update:model-value="set('format', String($event) as SaveConfig['format'])"
      />
    </div>
    <div v-if="cfg.format !== 'png'" class="si-row">
      <label class="si-lbl"><i class="mdi mdi-quality-high" /> Quality</label>
      <ZenNumber
        :model-value="cfg.quality"
        :min="1"
        :max="100"
        :step="1"
        :precision="0"
        @update:model-value="set('quality', Number($event))"
      />
    </div>

    <div class="si-sep" />

    <!-- ── metadata switches ─────────────────────────────────────── -->
    <div class="si-row">
      <label class="si-lbl" title="Embed the workflow + prompt inside saved files"
        ><i class="mdi mdi-database-outline" /> Embed metadata</label
      >
      <ZenSwitch
        :model-value="cfg.embed_metadata"
        on-icon="mdi mdi-check"
        @update:model-value="set('embed_metadata', !!$event)"
      />
    </div>
    <div class="si-row">
      <label class="si-lbl" title="Also write a sidecar .json with the full workflow"
        ><i class="mdi mdi-code-json" /> Workflow .json</label
      >
      <ZenSwitch
        :model-value="cfg.save_workflow_json"
        on-icon="mdi mdi-check"
        @update:model-value="set('save_workflow_json', !!$event)"
      />
    </div>

    <!-- ── advanced (collapsible) ────────────────────────────────── -->
    <button class="si-exp" @click="showAdv = !showAdv">
      <i class="mdi" :class="showAdv ? 'mdi-chevron-down' : 'mdi-chevron-right'" />
      <i class="mdi mdi-cog-outline" />
      <span>Advanced</span>
      <span v-if="cfg.base_directory.trim()" class="si-badge">set</span>
    </button>
    <div v-if="showAdv" class="si-exp-body">
      <label class="si-sub">Base directory override</label>
      <ZenInput
        :model-value="cfg.base_directory"
        placeholder="empty = Settings / output dir"
        sm
        @update:model-value="setBase(String($event))"
      />
    </div>

    <!-- ── destination footer ────────────────────────────────────── -->
    <div class="si-dest" :title="fullDest">
      <i class="mdi mdi-content-save-outline" />
      <span class="si-dest-path">{{ destLabel }}</span>
    </div>
  </ZenWidget>
</template>

<script lang="ts">
// Read by nodekit's `discoverWidgets`: the io type comes from this file's NAME
// (SaveInfoConfig.vue -> NYNXZ_SAVE_INFO_CONFIG, matching nodes/save/io_types.py), and these
// two exports carry what the type alone cannot.
import type { NodeDef, WidgetOptions } from "@/framework";
import { DEFAULT_CONFIG } from "@/lib/saveApi";

export const widgetOptions: WidgetOptions = {
  name: "config",
  minHeight: 200,
  // content-driven: the node grows/shrinks as the form expands (there is no image to fill)
  default: { ...DEFAULT_CONFIG },
};

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: "nynxz.IO.SaveInfo",
  minSize: [300, 300],
  // Middle-click the save_info output -> spawn + wire a Save Image.
  slotLinks: [{ output: "save_info", spawn: "nynxz.Save.Image" }],
};
</script>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";
import {
  ZenButton,
  ZenCombo,
  ZenIconButton,
  ZenInput,
  ZenNumber,
  ZenSwitch,
  ZenToggleGroup,
  ZenWidget,
  type ComboItem,
} from "@nynxz/zenkit-ui";
import { listProjects, normalizeConfig, type SaveConfig } from "@/lib/saveApi";

interface DOMWidget {
  name?: string;
  value: unknown;
  callback?: (v: unknown) => void;
}
const props = defineProps<{ widget?: DOMWidget; node?: unknown }>();

const cfg = ref<SaveConfig>(normalizeConfig(props.widget?.value));
const projects = ref<string[]>([]);
const baseDir = ref<string>("");
const showAdv = ref(cfg.value.base_directory.trim().length > 0);

const FORMAT_OPTS = [
  { value: "png", label: "PNG", icon: "mdi mdi-file-png-box" },
  { value: "jpg", label: "JPG", icon: "mdi mdi-file-jpg-box" },
  { value: "webp", label: "WEBP", icon: "mdi mdi-image-outline" },
];

// Write the whole object back through the widget so it serialises with the graph.
function commit() {
  if (!props.widget) return;
  const v = { ...cfg.value };
  // eslint-disable-next-line vue/no-mutating-props -- `widget` is ComfyUI's host object, not Vue state; writing its value IS how a node body persists.
  props.widget.value = v;
  try {
    props.widget.callback?.(v);
  } catch {
    /* no callback */
  }
}
function set<K extends keyof SaveConfig>(key: K, val: SaveConfig[K]) {
  cfg.value = { ...cfg.value, [key]: val };
  commit();
}

// --- project combo ---
const projectItems = computed<ComboItem[]>(() => {
  const items: ComboItem[] = [{ value: "", label: "output root" }];
  for (const p of projects.value) items.push({ value: p, label: p });
  // Keep a freshly-typed (not-yet-on-disk) project selectable.
  if (cfg.value.project && !projects.value.includes(cfg.value.project))
    items.push({ value: cfg.value.project, label: cfg.value.project });
  return items;
});
async function ensureProjects(force = false) {
  const r = await listProjects(cfg.value.base_directory, force);
  projects.value = r.projects;
  baseDir.value = r.base;
}

// --- inline "new project" create ---
const creating = ref(false);
const newName = ref("");
const newInput = ref<{ $el?: HTMLElement } | null>(null);
function cleanName(v: string) {
  return v
    .trim()
    .replace(/\\/g, "/")
    .replace(/^\/+|\/+$/g, "")
    .split("/")
    .filter((s) => s && s !== "..")
    .join("/");
}
function startCreate(close?: () => void) {
  close?.();
  newName.value = "";
  creating.value = true;
  nextTick(() => newInput.value?.$el?.querySelector("input")?.focus());
}
function confirmCreate() {
  const name = cleanName(newName.value);
  creating.value = false;
  if (!name) return;
  if (!projects.value.includes(name)) projects.value = [...projects.value, name];
  set("project", name); // folder is created on disk when a Save Image first runs
}
function cancelCreate() {
  creating.value = false;
}

// --- base dir override → relist projects against the new base ---
function setBase(v: string) {
  set("base_directory", v);
  ensureProjects(true);
}

// --- destination preview ---
const destLabel = computed(() =>
  [cfg.value.project || "output root", cfg.value.subfolder].filter(Boolean).join(" / "),
);
const fullDest = computed(() =>
  [baseDir.value || "output", cfg.value.project, cfg.value.subfolder].filter(Boolean).join("/"),
);

onMounted(() => {
  // Restore on graph load + keep tracking host writes (chained so we don't clobber).
  cfg.value = normalizeConfig(props.widget?.value);
  showAdv.value = showAdv.value || cfg.value.base_directory.trim().length > 0;
  ensureProjects();
  const prev = props.widget?.callback;
  if (props.widget) {
    // eslint-disable-next-line vue/no-mutating-props -- `widget` is ComfyUI's host object, not Vue state; writing its value IS how a node body persists.
    props.widget.callback = function (this: DOMWidget, v: unknown) {
      if (prev) {
        try {
          prev.call(this, v);
        } catch {
          /* ignore */
        }
      }
      cfg.value = normalizeConfig(v);
    };
  }
});
</script>

<style scoped>
/* content-driven panel: the node grows to fit this form */

/* stacked field (label above a full-width control) */
.si-field {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
/* inline row (label left, control right) */
.si-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-height: 24px;
}
.si-lbl {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
  font-weight: 600;
}
.si-lbl .mdi {
  font-size: 14px;
}

.si-grow {
  flex: 1;
  min-width: 0;
}
.si-project {
  display: flex;
  align-items: center;
  gap: 5px;
}

.si-sel {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
}
.si-sel .mdi {
  flex: none;
  font-size: 15px;
  color: var(--zen-muted, #9aa0aa);
}
.si-sel-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.si-opt-ic {
  flex: none;
  font-size: 15px;
  color: var(--zen-accent, #6366f1);
}
.si-opt-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.si-newbtn {
  justify-content: flex-start !important;
}

/* quality */
.si-quality {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  min-width: 0;
  max-width: 60%;
}
.si-qval {
  flex: none;
  min-width: 26px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--zen-text, #e5e5ea);
}

.si-sep {
  height: 1px;
  background: var(--zen-border, #34343c);
  opacity: 0.7;
  margin: 1px 0;
}

/* collapsible header */
.si-exp {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 3px 2px;
  background: none;
  border: none;
  cursor: pointer;
  font: inherit;
  font-size: 11px;
  font-weight: 600;
  color: var(--zen-muted, #9aa0aa);
  text-align: left;
}
.si-exp:hover {
  color: var(--zen-text, #e5e5ea);
}
.si-exp .mdi {
  font-size: 14px;
}
.si-badge {
  margin-left: auto;
  padding: 1px 6px;
  border-radius: 999px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--zen-accent-text, #fff);
  background: var(--zen-accent, #6366f1);
}
.si-exp-body {
  padding: 0 2px 2px;
}
.si-sub {
  display: block;
  margin-bottom: 3px;
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
}

/* destination footer */
.si-dest {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 1px;
  padding: 5px 7px;
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-input, #1b1b20);
  border: 1px solid var(--zen-border, #34343c);
  color: var(--zen-text, #e5e5ea);
  font-size: 11px;
}
.si-dest .mdi {
  flex: none;
  font-size: 14px;
  color: var(--zen-accent, #6366f1);
}
.si-dest-path {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
