<template>
  <ZenWidget :gap="6" pad="6px 4px" class="bp">
    <div class="bp-row">
      <label class="bp-lbl">Type</label>
      <ZenToggleGroup
        :model-value="cfg.bg_mode"
        :options="MODE_OPTS"
        @update:model-value="set('bg_mode', String($event) as BgStyle['bg_mode'])"
      />
    </div>

    <!-- SOLID -->
    <template v-if="cfg.bg_mode === 'solid'">
      <div class="bp-row">
        <label class="bp-lbl">Color</label>
        <ZenColorPicker
          :model-value="cfg.bg_color"
          @update:model-value="set('bg_color', String($event))"
        />
      </div>
    </template>

    <!-- GRADIENT -->
    <template v-else-if="cfg.bg_mode === 'gradient'">
      <div class="bp-row">
        <label class="bp-lbl">From → To</label>
        <div class="bp-pair">
          <ZenColorPicker
            :model-value="cfg.bg_color"
            @update:model-value="set('bg_color', String($event))"
          />
          <ZenColorPicker
            :model-value="cfg.bg_color2"
            @update:model-value="set('bg_color2', String($event))"
          />
        </div>
      </div>
      <div class="bp-row">
        <label class="bp-lbl">Angle</label>
        <ZenNumber
          :model-value="cfg.bg_angle"
          :min="0"
          :max="360"
          :step="1"
          :precision="0"
          @update:model-value="set('bg_angle', Number($event))"
        />
      </div>
    </template>

    <!-- PATTERN -->
    <template v-else-if="cfg.bg_mode === 'pattern'">
      <div class="bp-row">
        <label class="bp-lbl">Base / Tint</label>
        <div class="bp-pair">
          <ZenColorPicker
            :model-value="cfg.bg_color"
            @update:model-value="set('bg_color', String($event))"
          />
          <ZenColorPicker
            :model-value="cfg.pattern_color"
            @update:model-value="set('pattern_color', String($event))"
          />
        </div>
      </div>
      <div class="bp-patterns">
        <button
          v-for="p in PATTERNS"
          :key="p"
          class="bp-sw"
          :class="{ on: cfg.pattern === p }"
          :title="p"
          :style="{ background: cfg.bg_color }"
          @click="set('pattern', p)"
        >
          <span class="bp-sw-fill" :style="{ backgroundImage: patternCss(p, cfg.pattern_color) }" />
        </button>
      </div>
      <div class="bp-row">
        <label class="bp-lbl">Scale</label>
        <ZenNumber
          :model-value="cfg.pattern_scale"
          :min="6"
          :max="96"
          :step="1"
          :precision="0"
          @update:model-value="set('pattern_scale', Number($event))"
        />
      </div>
      <div class="bp-row">
        <label class="bp-lbl">Opacity</label>
        <ZenNumber
          :model-value="cfg.pattern_opacity"
          :min="0"
          :max="1"
          :step="0.05"
          :precision="2"
          @update:model-value="set('pattern_opacity', Number($event))"
        />
      </div>
    </template>

    <!-- TRANSPARENT -->
    <template v-else-if="cfg.bg_mode === 'transparent'">
      <p class="bp-hint">
        <i class="mdi mdi-checkerboard" /> See-through ground — saves a PNG/WebP with real
        transparency. For a glowing border, give each cell a coloured drop shadow on the Grid Style
        node (big blur, offset 0) so it blooms as a halo.
      </p>
    </template>

    <!-- ICONS -->
    <template v-else>
      <p class="bp-hint">
        <i class="mdi mdi-information-outline" /> Wire icon image(s) into the node — one tiles
        evenly, a batch shuffles by seed.
      </p>
      <div class="bp-row">
        <label class="bp-lbl">Base</label>
        <ZenColorPicker
          :model-value="cfg.bg_color"
          @update:model-value="set('bg_color', String($event))"
        />
      </div>
      <div class="bp-row">
        <label class="bp-lbl">Tint</label>
        <ZenSwitch
          :model-value="cfg.tint"
          on-icon="mdi mdi-check"
          @update:model-value="set('tint', !!$event)"
        />
      </div>
      <div v-if="cfg.tint" class="bp-row">
        <label class="bp-lbl">Tint color</label>
        <ZenColorPicker
          :model-value="cfg.tint_color"
          @update:model-value="set('tint_color', String($event))"
        />
      </div>
      <div class="bp-row">
        <label class="bp-lbl" title="Flip the icon's shape — for dark-on-light source icons"
          >Invert shape</label
        >
        <ZenSwitch
          :model-value="cfg.invert_alpha"
          on-icon="mdi mdi-check"
          @update:model-value="set('invert_alpha', !!$event)"
        />
      </div>
      <div class="bp-row">
        <label
          class="bp-lbl"
          title="How many icon columns across — the square cell size is sized to the output"
          >Columns</label
        >
        <ZenNumber
          :model-value="cfg.columns"
          :min="1"
          :max="64"
          :step="1"
          :precision="0"
          @update:model-value="set('columns', Number($event))"
        />
      </div>
      <div class="bp-row">
        <label class="bp-lbl">Icon scale</label>
        <ZenNumber
          :model-value="cfg.scale"
          :min="0.05"
          :max="4"
          :step="0.05"
          :precision="2"
          @update:model-value="set('scale', Number($event))"
        />
      </div>
      <div class="bp-row">
        <label class="bp-lbl">Opacity</label>
        <ZenNumber
          :model-value="cfg.opacity"
          :min="0"
          :max="1"
          :step="0.05"
          :precision="2"
          @update:model-value="set('opacity', Number($event))"
        />
      </div>
      <div class="bp-row">
        <label class="bp-lbl">Rotate ±</label>
        <ZenNumber
          :model-value="cfg.rotate_jitter"
          :min="0"
          :max="180"
          :step="1"
          :precision="0"
          @update:model-value="set('rotate_jitter', Number($event))"
        />
      </div>
      <div class="bp-row">
        <label class="bp-lbl">Size jitter</label>
        <ZenNumber
          :model-value="cfg.scale_jitter"
          :min="0"
          :max="1"
          :step="0.05"
          :precision="2"
          @update:model-value="set('scale_jitter', Number($event))"
        />
      </div>
    </template>
  </ZenWidget>
</template>

<script lang="ts">
// Read by nodekit's `discoverWidgets`: the io type comes from this file's NAME
// (BgConfig.vue -> NYNXZ_BG_CONFIG, matching nodes/background/io_types.py).
import type { NodeDef, WidgetOptions } from "@/framework";
// Aliased: the setup block below imports DEFAULT_BG too, and an SFC's two script blocks
// share one module scope.
import { DEFAULT_BG as BG_DEFAULTS } from "@/lib/bgApi";

export const widgetOptions: WidgetOptions = {
  name: "config",
  minHeight: 220,
  default: { ...BG_DEFAULTS },
};

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: "nynxz.Preview.Background",
};
</script>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ZenColorPicker, ZenNumber, ZenSwitch, ZenToggleGroup, ZenWidget } from "@nynxz/zenkit-ui";
import { DEFAULT_BG, normalizeBg, type BgStyle } from "@/lib/bgApi";
import { PATTERNS, patternCss } from "@/lib/bgPatterns";

interface DOMWidget {
  name?: string;
  value: unknown;
  callback?: (v: unknown) => void;
}
const props = defineProps<{ widget?: DOMWidget; node?: unknown }>();

const cfg = ref<BgStyle>(normalizeBg(props.widget?.value));
const MODE_OPTS = [
  { value: "solid", label: "Solid" },
  { value: "gradient", label: "Gradient" },
  { value: "pattern", label: "Pattern" },
  { value: "icons", label: "Icons" },
  { value: "transparent", label: "Transparent" },
];

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
function set<K extends keyof BgStyle>(key: K, val: BgStyle[K]) {
  cfg.value = { ...cfg.value, [key]: val };
  commit();
}

onMounted(() => {
  cfg.value = normalizeBg(props.widget?.value);
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
      cfg.value = normalizeBg(v);
    };
  }
});

void DEFAULT_BG;
</script>

<style scoped>
.bp-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-height: 24px;
}
.bp-lbl {
  flex: none;
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
  font-weight: 600;
}
.bp-pair {
  display: flex;
  align-items: center;
  gap: 5px;
  justify-content: flex-end;
}
.bp-hint {
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
.bp-hint .mdi {
  flex: none;
  font-size: 13px;
  color: var(--zen-accent, #6366f1);
}
.bp-patterns {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 5px;
  padding: 2px 0;
}
.bp-sw {
  position: relative;
  aspect-ratio: 1;
  padding: 0;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 6px);
  cursor: pointer;
  overflow: hidden;
}
.bp-sw:hover {
  border-color: var(--zen-accent, #6366f1);
}
.bp-sw.on {
  border-color: var(--zen-accent, #6366f1);
  box-shadow: 0 0 0 1px var(--zen-accent, #6366f1) inset;
}
.bp-sw-fill {
  position: absolute;
  inset: 0;
  background-repeat: repeat;
}
</style>
