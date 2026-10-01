<template>
  <ZenWidget :gap="2" pad="4px 4px 6px" class="gs">
    <!-- Geometry (columns / size / gaps / padding / fit) now lives in native node
             widgets so it can be wired in or promoted to a subgraph — see grid_style.py.
             This panel owns the decoration only. -->
    <!-- ── BORDER ─────────────────────────────────────────────────── -->
    <button class="gs-head" @click="open.border = !open.border">
      <i class="mdi" :class="open.border ? 'mdi-chevron-down' : 'mdi-chevron-right'" />
      <i class="mdi mdi-border-all-variant" /><span>Border</span>
      <span v-if="cfg.border_width > 0" class="gs-badge">{{ cfg.border_width }}px</span>
    </button>
    <div v-if="open.border" class="gs-body">
      <div class="gs-row">
        <label class="gs-lbl">Width</label>
        <ZenNumber
          :model-value="cfg.border_width"
          :min="0"
          :max="40"
          :step="1"
          :precision="0"
          @update:model-value="set('border_width', Number($event))"
        />
      </div>
      <template v-if="cfg.border_width > 0">
        <div class="gs-row">
          <label class="gs-lbl">Style</label>
          <ZenToggleGroup
            :model-value="cfg.border_mode"
            :options="BORDER_OPTS"
            @update:model-value="set('border_mode', String($event) as GridStyle['border_mode'])"
          />
        </div>
        <div v-if="cfg.border_mode === 'solid'" class="gs-row">
          <label class="gs-lbl">Color</label>
          <ZenColorPicker
            :model-value="cfg.border_color"
            @update:model-value="set('border_color', String($event))"
          />
        </div>
        <template v-else-if="cfg.border_mode === 'gradient'">
          <div class="gs-row">
            <label class="gs-lbl">From → To</label>
            <div class="gs-pair">
              <ZenColorPicker
                :model-value="cfg.border_color"
                @update:model-value="set('border_color', String($event))"
              />
              <ZenColorPicker
                :model-value="cfg.border_color2"
                @update:model-value="set('border_color2', String($event))"
              />
            </div>
          </div>
          <div class="gs-row">
            <label class="gs-lbl">Angle</label>
            <ZenNumber
              :model-value="cfg.border_angle"
              :min="0"
              :max="360"
              :step="1"
              :precision="0"
              @update:model-value="set('border_angle', Number($event))"
            />
          </div>
        </template>
        <p v-else class="gs-hint">
          <i class="mdi mdi-palette-outline" /> Each cell's border is sampled from its own image — a
          vivid colour that matches the content.
        </p>
      </template>
      <div class="gs-row">
        <label class="gs-lbl">Radius</label>
        <ZenNumber
          :model-value="cfg.border_radius"
          :min="0"
          :max="120"
          :step="1"
          :precision="0"
          @update:model-value="set('border_radius', Number($event))"
        />
      </div>
    </div>

    <!-- ── SHADOW ─────────────────────────────────────────────────── -->
    <button class="gs-head" @click="open.shadow = !open.shadow">
      <i class="mdi" :class="open.shadow ? 'mdi-chevron-down' : 'mdi-chevron-right'" />
      <i class="mdi" :class="cfg.shadow_mode === 'glow' ? 'mdi-blur-radial' : 'mdi-box-shadow'" />
      <span>{{ cfg.shadow_mode === "glow" ? "Glow" : "Shadow" }}</span>
      <span v-if="cfg.shadow" class="gs-badge">on</span>
    </button>
    <div v-if="open.shadow" class="gs-body">
      <div class="gs-row">
        <label class="gs-lbl">Enable</label>
        <ZenSwitch
          :model-value="cfg.shadow"
          on-icon="mdi mdi-check"
          @update:model-value="set('shadow', !!$event)"
        />
      </div>
      <template v-if="cfg.shadow">
        <div class="gs-row">
          <label class="gs-lbl">Style</label>
          <ZenToggleGroup
            :model-value="cfg.shadow_mode"
            :options="SHADOW_OPTS"
            @update:model-value="set('shadow_mode', String($event) as GridStyle['shadow_mode'])"
          />
        </div>
        <div class="gs-row">
          <label class="gs-lbl">{{ cfg.shadow_mode === "glow" ? "Size" : "Blur" }}</label>
          <ZenNumber
            :model-value="cfg.shadow_blur"
            :min="0"
            :max="80"
            :step="1"
            :precision="0"
            @update:model-value="set('shadow_blur', Number($event))"
          />
        </div>
        <div v-if="cfg.shadow_mode === 'drop'" class="gs-row">
          <label class="gs-lbl">Offset</label>
          <div class="gs-pair">
            <ZenNumber
              :model-value="cfg.shadow_offset_x"
              :min="-200"
              :max="200"
              :step="1"
              :precision="0"
              @update:model-value="set('shadow_offset_x', Number($event))"
            />
            <span class="gs-x">×</span>
            <ZenNumber
              :model-value="cfg.shadow_offset_y"
              :min="-200"
              :max="200"
              :step="1"
              :precision="0"
              @update:model-value="set('shadow_offset_y', Number($event))"
            />
          </div>
        </div>
        <div v-if="borderIsAuto" class="gs-row">
          <label class="gs-lbl" title="Tint the glow with each cell's own auto-border colour"
            >Match border</label
          >
          <ZenSwitch
            :model-value="cfg.shadow_match"
            on-icon="mdi mdi-check"
            @update:model-value="set('shadow_match', !!$event)"
          />
        </div>
        <div v-if="!(cfg.shadow_match && borderIsAuto)" class="gs-row">
          <label class="gs-lbl">Color</label>
          <ZenColorPicker
            :model-value="cfg.shadow_color"
            @update:model-value="set('shadow_color', String($event))"
          />
        </div>
        <div class="gs-row">
          <label class="gs-lbl">Opacity</label>
          <ZenNumber
            :model-value="cfg.shadow_opacity"
            :min="0"
            :max="1"
            :step="0.05"
            :precision="2"
            @update:model-value="set('shadow_opacity', Number($event))"
          />
        </div>
      </template>
    </div>
  </ZenWidget>
</template>

<script lang="ts">
// Read by nodekit's `discoverWidgets`: the io type comes from this file's NAME
// (GridStyleConfig.vue -> NYNXZ_GRID_STYLE_CONFIG, matching nodes/grid/io_types.py).
//
// Geometry lives in native widgets (declared in grid_style.py); this panel owns only the
// decoration (border + shadow).
import type { NodeDef, WidgetOptions } from "@/framework";
import { DEFAULT_GRID_DECO } from "@/lib/gridApi";

export const widgetOptions: WidgetOptions = {
  name: "config",
  minHeight: 220,
  // content-driven: the node grows as sections expand
  default: { ...DEFAULT_GRID_DECO },
};

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: "nynxz.Preview.GridStyle",
  minSize: [320, 360],
  slotLinks: [{ output: "style", spawn: "nynxz.Preview.GridFromImages" }],
};
</script>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ZenColorPicker, ZenNumber, ZenSwitch, ZenToggleGroup, ZenWidget } from "@nynxz/zenkit-ui";
import { gridDecoOnly, normalizeGridStyle, type GridStyle } from "@/lib/gridApi";

interface DOMWidget {
  name?: string;
  value: unknown;
  callback?: (v: unknown) => void;
}
const props = defineProps<{ widget?: DOMWidget; node?: unknown }>();

const cfg = ref<GridStyle>(normalizeGridStyle(props.widget?.value));
const open = ref({ border: true, shadow: false });

const BORDER_OPTS = [
  { value: "solid", label: "Solid" },
  { value: "gradient", label: "Gradient" },
  { value: "auto", label: "Auto" },
];
const SHADOW_OPTS = [
  { value: "drop", label: "Drop" },
  { value: "glow", label: "Glow" },
];
const borderIsAuto = computed(() => cfg.value.border_width > 0 && cfg.value.border_mode === "auto");

function commit() {
  if (!props.widget) return;
  // Persist decoration only — geometry is owned by the node's native widgets, so writing
  // it here would shadow them (execute treats stray geometry in `config` as a legacy override).
  const v = gridDecoOnly(cfg.value);
  // eslint-disable-next-line vue/no-mutating-props -- `widget` is ComfyUI's host object, not Vue state; writing its value IS how a node body persists.
  props.widget.value = v;
  try {
    props.widget.callback?.(v);
  } catch {
    /* no callback */
  }
}
function set<K extends keyof GridStyle>(key: K, val: GridStyle[K]) {
  cfg.value = { ...cfg.value, [key]: val };
  commit();
}

onMounted(() => {
  cfg.value = normalizeGridStyle(props.widget?.value);
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
      cfg.value = normalizeGridStyle(v);
    };
  }
});
</script>

<style scoped>
/* section header */
.gs-head {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 5px 2px;
  background: none;
  border: none;
  cursor: pointer;
  font: inherit;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--zen-muted, #9aa0aa);
  text-align: left;
}
.gs-head:hover {
  color: var(--zen-text, #e5e5ea);
}
.gs-head .mdi {
  font-size: 14px;
}
.gs-badge {
  margin-left: auto;
  padding: 1px 6px;
  border-radius: 999px;
  font-size: 9px;
  font-weight: 700;
  color: var(--zen-accent-text, #fff);
  background: var(--zen-accent, #6366f1);
}

.gs-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 2px 2px 8px 6px;
}
.gs-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-height: 24px;
}
.gs-lbl {
  flex: none;
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
  font-weight: 600;
}
.gs-pair {
  display: flex;
  align-items: center;
  gap: 5px;
  justify-content: flex-end;
}
.gs-x {
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
}
.gs-hint {
  display: flex;
  align-items: flex-start;
  gap: 5px;
  margin: 0;
  padding: 5px 7px;
  border-radius: var(--zen-radius, 6px);
  background: var(--zen-input, #1b1b20);
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
  line-height: 1.35;
}
.gs-hint .mdi {
  flex: none;
  font-size: 13px;
  color: var(--zen-accent, #6366f1);
}
</style>
