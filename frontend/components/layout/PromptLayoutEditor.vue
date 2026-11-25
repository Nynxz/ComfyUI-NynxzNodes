<template>
  <div class="pl-host" @keydown="onKey" @contextmenu="keepContextMenu">
    <LayoutEditor
      ref="editor"
      v-model:selected="selected"
      v-model:tool="tool"
      :mode="mode"
      :modes="PROMPT_MODES"
      :layout="layout"
      :node="node"
      :in-panel="inPanel"
      property-prefix="nynxzLayout"
      :help="HELP"
      :rail-active="wand"
      :mask-view="maskView"
      :mask-clips="maskClips"
      accepts-drops
      :tabs="[{ id: 'mixins', label: 'Mixins', count: scene.mixins.length }]"
      @pick-tool="wand = false"
      @pop-out="emit('popOut')"
      @open="openBox"
      @drop="onDrop"
    >
      <template #box="{ box: face, maskStyle }">
        <LayoutBoxFace
          v-if="showFaces && boxById.get(face.id)"
          :scene="scene"
          :box="boxById.get(face.id)!"
          :mask-style="maskStyle"
        />
        <div v-else-if="faceMixins(face.id).length" class="pl-face">
          <MixinAvatar
            v-for="mixin in faceMixins(face.id)"
            :key="mixin.id"
            :mixin="mixin"
            :size="18"
            :title="mixin.name"
          />
        </div>
      </template>

      <template #overlay>
        <div v-if="wand" class="pl-wand" @pointerdown.stop.prevent="onWandClick" />
        <template v-if="box">
          <span
            v-for="(pt, i) in box.points"
            :key="i"
            class="pl-point"
            :class="{ out: !pt.include, live: wand }"
            :style="pointStyle(pt, i)"
            :title="wand ? 'Drag to move · click to remove · Alt+click to flip' : undefined"
            @pointerdown.stop.prevent="onPointDown($event, i)"
            @pointermove="onPointMove"
            @pointerup="onPointUp($event, i)"
            @pointercancel="pointDrag = null"
          />
        </template>
      </template>

      <template #tab-mixins>
        <LayoutMixins
          v-model:open="openMixin"
          :boxes="scene.boxes"
          :selected-boxes="selected"
          :model-value="scene.mixins"
          @update:model-value="setMixins"
          @create="addMixin()"
          @select-boxes="selectMixinBoxes"
          @assign-selected="assignMixinToSelected"
          @open-library="openMixinLibrary($event)"
          @swap-from-library="openMixinSwap($event)"
          @swap-boxes="swapMixinBoxes"
        />
      </template>

      <template #add>
        <ZenButton
          v-if="scene.mode !== 'regions'"
          sm
          variant="ghost"
          icon="mdi mdi-format-text"
          title="Add a text box — words the picture renders"
          @click="addText"
        >
          Text
        </ZenButton>
      </template>

      <template #rail>
        <ZenIconButton
          size="sm"
          icon="mdi mdi-auto-fix"
          :active="wand"
          title="Select with SAM3 — click what to capture, Alt+click what to leave out (W). Needs a photo and SAM3 on Layout Inputs."
          @click="setWand(!wand)"
        />
        <ZenIconButton
          size="sm"
          :icon="showFaces ? 'mdi mdi-card-text-outline' : 'mdi mdi-card-off-outline'"
          :active="showFaces"
          title="Show each box's references, LoRAs and prompt on the canvas"
          @click="setShowFaces(!showFaces)"
        />
        <ZenButton
          sm
          variant="ghost"
          icon="mdi mdi-creation-outline"
          title="Layout agent — a model draws the boxes from the trace image or your words (needs Layout Inputs)"
          @click="agentOpen = true"
        >
          Agent
        </ZenButton>
      </template>

      <template #status>
        <ZenPopover placement="top-end" role="dialog">
          <template #trigger="{ toggle }">
            <button
              type="button"
              class="pl-status"
              :class="{ warn: warnings.length }"
              :title="
                warnings.length
                  ? warnings.join('\n')
                  : 'The prompt the node outputs — click to see and copy it'
              "
              @click="toggle"
            >
              <i
                :class="
                  compiling
                    ? 'mdi mdi-loading mdi-spin'
                    : warnings.length
                      ? 'mdi mdi-alert-outline'
                      : 'mdi mdi-check'
                "
              />
              {{ warnings.length ? `${warnings.length} to check` : "Prompt" }}
              <span class="pl-count">{{ output.length }} chars</span>
            </button>
          </template>
          <div class="pl-pop">
            <div class="pl-pop-head">
              <span class="pl-pop-title">{{ mode.label }} prompt</span>
              <ZenButton sm icon="mdi mdi-content-copy" :disabled="!output" @click="copyOutput">
                {{ copied ? "Copied" : "Copy" }}
              </ZenButton>
            </div>
            <pre class="pl-out">{{ output || "—" }}</pre>
            <ul v-if="warnings.length" class="pl-warn">
              <li v-for="w in warnings" :key="w">{{ w }}</li>
            </ul>
          </div>
        </ZenPopover>
      </template>

      <template #foot>
        <ZenButton
          sm
          variant="ghost"
          :icon="mode.icon"
          title="The mode this layout is for — click to change it"
          @click="settingsOpen = true"
        >
          {{ mode.label }}
          <i class="mdi mdi-chevron-up pl-cog" />
        </ZenButton>
      </template>

      <template #element-actions>
        <ZenPopover v-if="scene.boxes.length > 1 && box" placement="bottom-end" role="dialog">
          <template #trigger="{ toggle }">
            <ZenIconButton
              size="sm"
              icon="mdi mdi-swap-horizontal"
              title="Swap content with another box — description, mixins, references and LoRAs trade places; each keeps its position and shape"
              @click="toggle"
            />
          </template>
          <template #default="{ close }">
            <div class="pl-menu">
              <span class="pl-menu-title">Swap content with</span>
              <ZenButton
                v-for="other in scene.boxes.filter((b) => b.id !== box!.id)"
                :key="other.id"
                sm
                @click="swapWith(other.id, close)"
              >
                <span class="pl-swatch" :style="{ background: other.color }" />
                {{ other.name }}
              </ZenButton>
            </div>
          </template>
        </ZenPopover>
      </template>

      <template #element>
        <div v-if="box" class="pl-form">
          <FrescoSection label="Mixins" :note="mixinNote">
            <template #actions>
              <ZenIconButton
                v-if="canPromote"
                size="sm"
                icon="mdi mdi-share-variant-outline"
                title="Promote this box's description, references and LoRAs into a reusable mixin"
                @click="promoteToMixin"
              />
              <ZenIconButton
                size="sm"
                icon="mdi mdi-bookshelf"
                title="Open Mixin Library — browse, create or add a saved mixin"
                @click="openMixinLibrary()"
              />
              <ZenIconButton
                size="sm"
                icon="mdi mdi-plus"
                title="New mixin, in this box"
                @click="addMixin(box.name)"
              />
              <ZenIconButton
                size="sm"
                icon="mdi mdi-pencil-outline"
                title="Edit mixins"
                @click="editMixin(box.mixins[0])"
              />
            </template>
            <div v-if="scene.mixins.length" class="pl-chips">
              <button
                v-for="mixin in scene.mixins"
                :key="mixin.id"
                type="button"
                :class="['pl-chip', { on: box.mixins.includes(mixin.id) }]"
                :title="chipTooltip(mixin, box.mixins.includes(mixin.id))"
                @click="toggleMixin(mixin.id)"
                @dblclick="editMixin(mixin.id)"
                @contextmenu.prevent.stop="openChipMenu($event, mixin)"
              >
                <MixinAvatar :mixin="mixin" :size="18" />
                {{ mixin.name }}
              </button>
            </div>
            <p v-else class="pl-hint">
              None yet — + makes a reusable mixin for shared characters/styles.
            </p>
          </FrescoSection>

          <FrescoSection v-if="scene.mode !== 'regions'" label="Kind">
            <ZenToggleGroup
              class="pl-seg"
              size="sm"
              :model-value="box.kind"
              :options="kindOptions"
              @update:model-value="patchBox({ kind: $event as BoxKind })"
            />
          </FrescoSection>

          <FrescoSection v-if="box.kind === 'text'" label="Text" note="rendered exactly">
            <ZenInput
              sm
              :model-value="box.text"
              placeholder="The words to render"
              @update:model-value="patchBox({ text: String($event ?? '') })"
            />
          </FrescoSection>

          <FrescoSection label="Description" :note="citeNote">
            <template v-if="citeId" #actions>
              <ZenIconButton
                size="sm"
                icon="mdi mdi-at"
                :title="`Add <${citeId}> to the end of the prompt`"
                @click="cite(citeId)"
              />
            </template>
            <ZenInput
              ref="descInput"
              type="textarea"
              :rows="4"
              sm
              :model-value="box.desc"
              :placeholder="descPlaceholder"
              @update:model-value="patchBox({ desc: String($event ?? '') })"
            />
          </FrescoSection>

          <FrescoSection label="References" :note="boxRefNote">
            <template #actions>
              <ZenIconButton
                size="sm"
                icon="mdi mdi-upload-outline"
                title="Upload reference images for this box"
                :disabled="boxRefsPanel?.busy"
                @click="boxRefsPanel?.upload()"
              />
              <ZenIconButton
                size="sm"
                icon="mdi mdi-view-grid-plus-outline"
                title="Browse images for this box"
                @click="boxRefsPanel?.openBrowse()"
              />
            </template>
            <div v-if="inheritedRefs.length" class="pl-inherited">
              <button
                v-for="item in inheritedRefs"
                :key="item.ref.id"
                type="button"
                class="pl-inh-ref"
                :title="`From mixin ${item.mixin.name} — click to edit mixin`"
                @click="editMixin(item.mixin.id)"
              >
                <img :src="thumbUrl(item.ref.ref, item.ref.type)" alt="" />
                <span>{{ item.mixin.name }}</span>
              </button>
            </div>
            <FrescoRefs
              ref="boxRefsPanel"
              :model-value="box.refs"
              @update:model-value="patchBox({ refs: $event })"
            />
            <div v-if="totalActiveRefs > 1" class="pl-blend-row">
              <span :title="BLEND_TIP">blend</span>
              <ZenSelect
                size="sm"
                :model-value="box.blend"
                :options="BLENDS"
                :title="BLEND_TIP"
                @update:model-value="patchBox({ blend: $event as Blend })"
              />
              <ZenNumber
                class="pl-soft"
                size="sm"
                :model-value="box.softness"
                :min="0"
                :max="1"
                :step="0.05"
                :precision="2"
                title="Softness between references — 0 hard cells, 1 fully feathered"
                @update:model-value="patchBox({ softness: $event })"
              />
            </div>
          </FrescoSection>

          <FrescoSection label="LoRAs" :note="boxLoraNote">
            <template #actions>
              <ZenIconButton
                size="sm"
                icon="mdi mdi-plus"
                title="Add a LoRA just for this box"
                @click="addBoxLora"
              />
            </template>
            <div v-if="inheritedLoras.length" class="pl-inherited">
              <button
                v-for="item in inheritedLoras"
                :key="item.lora.id"
                type="button"
                class="pl-inh-lora"
                :title="`From mixin ${item.mixin.name} — click to edit mixin`"
                @click="editMixin(item.mixin.id)"
              >
                <i class="mdi mdi-puzzle-outline" />
                <span class="pl-inh-name">{{ shortLora(item.lora.name) }}</span>
                <span class="pl-inh-str">{{ item.lora.strength.toFixed(2) }}</span>
              </button>
            </div>
            <FrescoLoras
              v-if="box.loras.length"
              :model-value="box.loras"
              @update:model-value="patchBox({ loras: $event })"
            />
          </FrescoSection>

          <FrescoSection label="Capture" :note="captureNote">
            <template #actions>
              <ZenIconButton
                v-if="box.points.length"
                size="sm"
                icon="mdi mdi-close-circle-outline"
                title="Clear the click points"
                @click="patchBox({ points: [] })"
              />
              <ZenIconButton
                size="sm"
                icon="mdi mdi-auto-fix"
                :disabled="segmenting || (!box.capture.trim() && !box.points.length)"
                title="Segment now with SAM3 (Layout runs on its own)"
                @click="segment"
              />
            </template>
            <ZenInput
              sm
              :model-value="box.capture"
              placeholder="e.g. jacket, -hands — or click with the wand (W)"
              @update:model-value="
                segmentError = '';
                patchBox({ capture: String($event ?? '') });
              "
              @keydown.enter="segment"
            />
            <p v-if="segmentError" class="pl-seg-err">
              <i class="mdi mdi-alert-circle-outline" />
              {{ segmentError }}
            </p>
          </FrescoSection>

          <FrescoSection
            label="Shape"
            :note="box.mask ? 'painted · B paint, E erase' : 'whole box · B to paint'"
          >
            <template #actions>
              <ZenIconButton
                v-if="box.mask"
                size="sm"
                icon="mdi mdi-crop-square"
                title="Clear the shape — the box becomes its whole rectangle"
                @click="patchBox({ mask: null })"
              />
            </template>
            <div class="pl-pairs">
              <label class="pl-pair">
                <span :title="GROW_TIP">grow</span>
                <ZenNumber
                  size="sm"
                  :model-value="box.grow"
                  :min="-256"
                  :max="256"
                  :step="1"
                  :precision="0"
                  :title="GROW_TIP"
                  @update:model-value="patchBox({ grow: Math.round($event) })"
                />
              </label>
              <label class="pl-pair">
                <span title="Soften the edge, in canvas pixels">feather</span>
                <ZenNumber
                  size="sm"
                  :model-value="box.feather"
                  :min="0"
                  :max="256"
                  :step="1"
                  :precision="0"
                  title="Soften the edge, in canvas pixels"
                  @update:model-value="patchBox({ feather: Math.round($event) })"
                />
              </label>
            </div>
            <div v-if="box.mask" class="pl-grid">
              <span :title="SHAPE_TIP">shape for</span>
              <FrescoPills
                :model-value="box.shapeFor"
                :options="SHAPE_FOR"
                @update:model-value="patchBox({ shapeFor: $event as ShapeFor })"
              />
            </div>
          </FrescoSection>

          <FrescoSection label="Strength">
            <div class="pl-pairs">
              <label class="pl-pair">
                <span :title="INFLUENCE_TIP">influence</span>
                <ZenNumber
                  size="sm"
                  :model-value="box.influence"
                  :min="0"
                  :max="4"
                  :step="0.05"
                  :precision="2"
                  :title="INFLUENCE_TIP"
                  @update:model-value="patchBox({ influence: $event })"
                />
              </label>
              <label class="pl-pair">
                <span :title="DENOISE_TIP">denoise</span>
                <ZenNumber
                  size="sm"
                  :model-value="box.denoise"
                  :min="0"
                  :max="1"
                  :step="0.05"
                  :precision="2"
                  :title="DENOISE_TIP"
                  @update:model-value="patchBox({ denoise: $event })"
                />
              </label>
            </div>
          </FrescoSection>

          <FrescoSection
            v-if="scene.mode === 'ideogram' || scene.mode === 'moss'"
            label="Colours"
            note="up to 5"
          >
            <PaletteEdit
              :model-value="box.palette"
              @update:model-value="patchBox({ palette: $event })"
            />
          </FrescoSection>

          <FrescoSection v-if="scene.mode === 'moss'" label="Tags" note="comma separated">
            <ZenInput
              sm
              :model-value="box.tags.join(', ')"
              placeholder="e.g. 1girl, red scarf"
              @update:model-value="patchBox({ tags: splitList($event) })"
            />
          </FrescoSection>
        </div>
      </template>

      <template #multi="{ ids }">
        <div class="pl-form">
          <p class="pl-hint">
            Drag any of them on the canvas to move them together; the frame's handles scale them.
          </p>
          <FrescoSection label="Mixins" :note="`apply to all ${ids.length} boxes`">
            <div v-if="scene.mixins.length" class="pl-chips">
              <button
                v-for="mixin in scene.mixins"
                :key="mixin.id"
                type="button"
                :class="[
                  'pl-chip',
                  {
                    on: multiMixinState(ids, mixin.id) === 'all',
                    partial: multiMixinState(ids, mixin.id) === 'some',
                  },
                ]"
                :title="`Toggle ${mixin.name} across all ${ids.length} selected boxes`"
                @click="toggleMultiMixin(ids, mixin.id)"
              >
                <MixinAvatar :mixin="mixin" :size="18" />
                {{ mixin.name }}
              </button>
            </div>
            <p v-else class="pl-hint">Create a mixin in the Mixins tab to assign across boxes.</p>
          </FrescoSection>
          <FrescoSection label="Batch strength" :note="`set for all ${ids.length} boxes`">
            <div class="pl-pairs">
              <label class="pl-pair">
                <span :title="INFLUENCE_TIP">influence</span>
                <ZenNumber
                  size="sm"
                  :model-value="multiAverage(ids, 'influence')"
                  :min="0"
                  :max="4"
                  :step="0.05"
                  :precision="2"
                  @update:model-value="patchSelectedBoxes(ids, { influence: $event })"
                />
              </label>
              <label class="pl-pair">
                <span :title="DENOISE_TIP">denoise</span>
                <ZenNumber
                  size="sm"
                  :model-value="multiAverage(ids, 'denoise')"
                  :min="0"
                  :max="1"
                  :step="0.05"
                  :precision="2"
                  @update:model-value="patchSelectedBoxes(ids, { denoise: $event })"
                />
              </label>
            </div>
          </FrescoSection>
        </div>
      </template>

      <template #scene>
        <div class="pl-form">
          <FrescoSection :label="PROMPT_LABEL[scene.mode]" :note="PROMPT_NOTE[scene.mode]">
            <ZenInput
              type="textarea"
              :rows="4"
              sm
              :model-value="scene.prompt"
              placeholder="The whole picture"
              @update:model-value="patchScene({ prompt: String($event ?? '') })"
            />
          </FrescoSection>

          <FrescoSection
            v-if="scene.mode === 'ideogram'"
            label="Background"
            note="behind every box"
          >
            <ZenInput
              type="textarea"
              :rows="2"
              sm
              :model-value="scene.background"
              placeholder="What's behind every box"
              @update:model-value="patchScene({ background: String($event ?? '') })"
            />
          </FrescoSection>

          <FrescoSection v-if="scene.mode === 'ideogram'" label="Style" note="style_description">
            <ZenToggleGroup
              class="pl-seg"
              size="sm"
              :model-value="scene.style.on ? scene.style.kind : 'off'"
              :options="STYLE_KINDS"
              @update:model-value="setStyleKind(String($event))"
            />
            <template v-if="scene.style.on">
              <ZenField v-for="f in styleFields" :key="f.key" :label="f.label" stack>
                <PresetInput
                  :model-value="scene.style[f.key]"
                  :placeholder="f.hint"
                  :presets="STYLE_PRESETS[f.key]"
                  @update:model-value="patchStyle({ [f.key]: $event })"
                />
              </ZenField>
              <ZenField label="Palette" stack hint="Up to 16. Right-click a colour to drop it.">
                <PaletteEdit
                  :model-value="scene.style.palette"
                  :max="16"
                  @update:model-value="patchStyle({ palette: $event })"
                />
              </ZenField>
            </template>
          </FrescoSection>

          <template v-if="scene.mode === 'moss'">
            <FrescoSection label="Meta" note="what kind of image">
              <ZenInput
                sm
                :model-value="scene.meta.join(', ')"
                placeholder="poster, illustration"
                @update:model-value="patchScene({ meta: splitList($event) })"
              />
            </FrescoSection>
            <FrescoSection label="Layout" note="one picture, panels or a collage">
              <ZenSelect
                size="sm"
                :model-value="scene.canvas"
                :options="CANVAS_OPTIONS"
                @update:model-value="patchScene({ canvas: $event as MossCanvas })"
              />
            </FrescoSection>
            <FrescoSection label="Palette" note="up to 16">
              <PaletteEdit
                :model-value="scene.palette"
                :max="16"
                @update:model-value="patchScene({ palette: $event })"
              />
            </FrescoSection>
          </template>

          <p v-if="scene.mode === 'klein'" class="pl-note">
            Klein reads only the caption and each box's description, so colours go in the words.
            Cite every box in the caption as &lt;obj_N&gt;.
          </p>

          <FrescoSection label="Regions & references">
            <div class="pl-grid" title="How overlapping boxes share the pixels they both cover">
              <span>overlap</span>
              <FrescoPills
                :model-value="scene.settings.overlap"
                :options="OVERLAPS"
                @update:model-value="patchSettings({ overlap: $event as Overlap })"
              />
            </div>
            <div
              class="pl-grid"
              title="Resolution reference images are read at in Layout Apply. Higher reads more detail and costs more tokens."
            >
              <span>ref detail</span>
              <ZenNumber
                size="sm"
                :model-value="scene.settings.visual_size"
                :min="128"
                :max="1024"
                :step="32"
                @update:model-value="patchSettings({ visual_size: $event })"
              />
            </div>
          </FrescoSection>

          <FrescoSection label="Background region" note="wherever no box is">
            <template #actions>
              <ZenDot
                :model-value="scene.backdrop.on"
                title="Enable background region (for edits or regional backdrop prompt/LoRAs)"
                @update:model-value="patchBackdrop({ on: $event })"
              />
            </template>
            <template v-if="scene.backdrop.on">
              <ZenInput
                type="textarea"
                :rows="2"
                sm
                :model-value="scene.backdrop.prompt"
                placeholder="Optional — what fills the area outside every box"
                @update:model-value="patchBackdrop({ prompt: String($event ?? '') })"
              />
              <FrescoRefs
                :model-value="scene.backdrop.refs"
                @update:model-value="patchBackdrop({ refs: $event })"
              />
              <div class="pl-row-head">
                <span class="pl-sub">Background LoRAs</span>
                <ZenIconButton
                  size="sm"
                  icon="mdi mdi-plus"
                  title="Add a LoRA for the background"
                  @click="addBackdropLora"
                />
              </div>
              <FrescoLoras
                v-if="scene.backdrop.loras.length"
                :model-value="scene.backdrop.loras"
                @update:model-value="patchBackdrop({ loras: $event })"
              />
              <div class="pl-pairs">
                <label class="pl-pair">
                  <span :title="INFLUENCE_TIP">influence</span>
                  <ZenNumber
                    size="sm"
                    :model-value="scene.backdrop.influence"
                    :min="0"
                    :max="4"
                    :step="0.05"
                    :precision="2"
                    @update:model-value="patchBackdrop({ influence: $event })"
                  />
                </label>
                <label class="pl-pair">
                  <span :title="DENOISE_TIP">denoise</span>
                  <ZenNumber
                    size="sm"
                    :model-value="scene.backdrop.denoise"
                    :min="0"
                    :max="1"
                    :step="0.05"
                    :precision="2"
                    @update:model-value="patchBackdrop({ denoise: $event })"
                  />
                </label>
              </div>
            </template>
          </FrescoSection>
        </div>
      </template>
    </LayoutEditor>

    <LayoutAgent
      v-model:open="agentOpen"
      :scene="scene"
      :node-id="String(node?.id ?? '')"
      :image="editor?.backdrop ?? null"
      @apply="applyAgent"
    />
    <LayoutSettings
      v-model:open="settingsOpen"
      :mode="scene.mode"
      @pick="patchScene({ mode: $event })"
    />
    <MixinLibraryDialog
      v-model:open="mixinLibOpen"
      :layout-mixins="scene.mixins"
      :selected-boxes="selected"
      :swap-target="mixinSwapTarget"
      :initial-mixin="mixinLibInitial"
      @add="addMixinFromLibrary"
      @swap="swapMixinFromLibrary"
      @assign-existing="assignMixinToSelected"
      @sync-from-library="syncMixinFromLibrary"
      @link-layout-mixin="linkLayoutMixin"
    />
    <ZenContextMenu ref="chipMenu" :items="chipMenuItems" />
    <div v-if="choosing" class="pl-choose">
      <div class="pl-choose-card">
        <span class="pl-choose-title">What is this layout for?</span>
        <span class="pl-choose-note">
          Pick a starting mode. The mode button at the bottom left switches any time — every box,
          painted shape, mixin, reference and LoRA stays.
        </span>
        <ModeCards :model-value="null" @update:model-value="pickMode" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * The unified Layout node's editor: combines native layout prompt modes (MOSS, Ideogram, Klein),
 * plain Regions mode, reusable Mixins, direct per-box references & LoRAs, painted shapes, SAM3
 * wand/capture, drag-and-drop images on canvas, box-face previews, and background regions.
 */
import { api } from "@comfy/api";
import { app } from "@comfy/app";
import {
  ZenButton,
  ZenContextMenu,
  ZenDot,
  ZenField,
  ZenIconButton,
  ZenInput,
  ZenNumber,
  ZenPopover,
  ZenSelect,
  ZenToggleGroup,
  boxAt,
  type BoxRect,
  type BoxTool,
  type ContextMenuItem,
} from "@nynxz/zenkit-ui";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch, watchEffect } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoPills from "@/components/fresco/FrescoPills.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import LayoutAgent from "@/components/layout/LayoutAgent.vue";
import LayoutBoxFace from "@/components/layout/LayoutBoxFace.vue";
import LayoutEditor from "@/components/layout/LayoutEditor.vue";
import LayoutMixins from "@/components/layout/LayoutMixins.vue";
import LayoutSettings from "@/components/layout/LayoutSettings.vue";
import MixinAvatar from "@/components/layout/MixinAvatar.vue";
import MixinLibraryDialog from "@/components/layout/MixinLibraryDialog.vue";
import ModeCards from "@/components/layout/ModeCards.vue";
import PaletteEdit from "@/components/layout/PaletteEdit.vue";
import PresetInput from "@/components/layout/PresetInput.vue";
import { readMediaDrop } from "@/framework";
import type { AgentImage, AgentLayout } from "@/layout/agent";
import { compileLayout, saveMixinPreset } from "@/layout/library";
import { PROMPT_MODES } from "@/layout/modes";
import {
  boxActiveRefs,
  boxId,
  kleinId,
  nextColor,
  readBox,
  readMixin,
  swapBoxContent,
  type BoxKind,
  type IdeogramStyle,
  type LayoutBackdrop,
  type LayoutRegionSettings,
  type Mixin,
  type MossCanvas,
  type PromptBox,
  type PromptModeId,
  type PromptScene,
  type ShapeFor,
} from "@/layout/promptScene";
import type { SharedLayout } from "@/layout/store";
import { STYLE_PRESETS } from "@/layout/stylePresets";
import {
  BLENDS,
  isImage,
  newRef,
  refFromMedia,
  type Blend,
  type FrescoLora,
  type FrescoPoint,
  type FrescoRef,
  type Overlap,
} from "@/lib/frescoScene";
import { growMask } from "@/lib/maskGrow";
import { thumbUrl } from "@/lib/mediaRefs";
import { keepContextMenu } from "@/lib/ownContextMenu";
import { uid } from "@/lib/sceneUtil";
import { useSentBack, type SentImage } from "@/lib/sentBack";

const props = defineProps<{
  layout: SharedLayout<PromptScene>;
  node?: {
    id?: string | number;
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
  };
  inPanel?: boolean;
  /** A new layout: ask which mode it's for before anything else. */
  chooseMode?: boolean;
}>();
const emit = defineEmits<{ popOut: [] }>();

const HELP: [string, string][] = [
  ["Order", "bottom to top — background first, foreground on top"],
  ["Hidden boxes", "left out of the prompt and regions"],
  ["W", "SAM3 wand — click what to capture, Alt+click to leave out"],
  ["Drop an image", "on a box: add reference · on empty canvas: new box"],
];
const PROMPT_LABEL: Record<PromptModeId, string> = {
  regions: "Scene prompt",
  ideogram: "High-level description",
  klein: "Caption",
  moss: "Caption",
};
const PROMPT_NOTE: Record<PromptModeId, string> = {
  regions: "the whole picture (positive)",
  ideogram: "the whole picture",
  klein: "cite each box as <obj_N>",
  moss: "the scene and its style",
};
const GROW_TIP =
  "Grow the shape by this many canvas pixels, or shrink it (negative) — within the box.";
const SHAPE_TIP =
  "Whole box: the painted shape is the region. References: the region is its whole box (prompt and LoRAs everywhere) and the shape limits only where its references are read.";
const SHAPE_FOR = [
  { value: "layer", label: "Whole box", title: "The painted shape is the region" },
  {
    value: "refs",
    label: "References",
    title: "The box stays whole; the shape limits only its references",
  },
];
const OVERLAPS = [
  { value: "top", label: "Cover", title: "Upper boxes cover lower ones, like paint" },
  { value: "share", label: "Share", title: "Overlapping boxes split the overlap" },
];
const BLEND_TIP =
  "How multiple references combine: Mix interleaves evenly, Weighted follows strengths, Tiled keeps parts apart, Salient picks what shows something there.";
const INFLUENCE_TIP =
  "How strongly this region follows its own prompt and references against the rest of the picture.";
const DENOISE_TIP = "How much of a source photo this box may rewrite when editing in Layout Apply.";

const editor = ref<{
  addBox: (rect: BoxRect) => string;
  update: (next: unknown) => void;
  offerBackdrop: (image: SentImage) => void;
  backdrop: AgentImage | null;
  showTab: (id: string) => void;
} | null>(null);
const descInput = ref<{ $el?: HTMLElement } | null>(null);
const boxRefsPanel = ref<{ upload: () => void; openBrowse: () => void; busy: boolean } | null>(
  null,
);
const agentOpen = ref(false);
const selected = ref<string[]>([]);
const tool = ref<BoxTool>("select");
const settingsOpen = ref(false);
const openMixin = ref<string | null>(null);
const choosing = ref(!!props.chooseMode);
const scene = props.layout.scene;
const mode = computed(
  () => PROMPT_MODES.find((m) => m.id === scene.value.mode) ?? PROMPT_MODES[0]!,
);
const boxById = computed(() => new Map(scene.value.boxes.map((b) => [b.id, b])));

function update(next: PromptScene) {
  editor.value?.update(next);
}
function patchScene(patch: Partial<PromptScene>) {
  update({ ...scene.value, ...patch });
}
function patchSettings(patch: Partial<LayoutRegionSettings>) {
  patchScene({ settings: { ...scene.value.settings, ...patch } });
}
function patchBackdrop(patch: Partial<LayoutBackdrop>) {
  patchScene({ backdrop: { ...scene.value.backdrop, ...patch } });
}
function addBackdropLora() {
  patchBackdrop({
    loras: [...scene.value.backdrop.loras, { id: uid("o"), name: "", strength: 1, on: true }],
  });
}
function addText() {
  const id = editor.value?.addBox({ x: 0.2, y: 0.08, w: 0.6, h: 0.14 });
  if (id)
    patchScene({ boxes: scene.value.boxes.map((b) => (b.id === id ? { ...b, kind: "text" } : b)) });
}
function pickMode(id: PromptModeId) {
  choosing.value = false;
  patchScene({
    mode: id,
    ...(id === "regions" ? { backdrop: { ...scene.value.backdrop, on: true } } : {}),
  });
}
const splitList = (v: unknown) =>
  String(v ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

// --- display preference: canvas box previews ----------------------------------------------------

const FACES_KEY = "nynxzLayoutFaces";
const showFaces = ref(props.node?.properties?.[FACES_KEY] !== false);
function setShowFaces(on: boolean) {
  showFaces.value = on;
  if (props.node?.properties) {
    // eslint-disable-next-line vue/no-mutating-props -- node.properties persists UI preferences
    props.node.properties[FACES_KEY] = on;
  }
}

// --- what the whole scene reads, per mode -------------------------------------------------------

const STYLE_KINDS = [
  { value: "off", label: "Off" },
  { value: "photo", label: "Photo", icon: "mdi mdi-camera-outline" },
  { value: "art", label: "Art", icon: "mdi mdi-palette-outline" },
];
const CANVAS_OPTIONS = [
  { value: "", label: "Not said" },
  { value: "single", label: "Single image" },
  { value: "multi_panel", label: "Multi-panel" },
  { value: "collage", label: "Collage" },
];
type StyleText = "aesthetics" | "lighting" | "medium" | "photo" | "art_style";
const styleFields = computed<{ key: StyleText; label: string; hint: string }[]>(() => [
  { key: "aesthetics", label: "Aesthetics", hint: "moody, minimal, cinematic" },
  { key: "lighting", label: "Lighting", hint: "soft window light, neon" },
  { key: "medium", label: "Medium", hint: "photograph, oil painting" },
  scene.value.style.kind === "photo"
    ? { key: "photo", label: "Photo", hint: "35mm, shallow depth of field" }
    : { key: "art_style", label: "Art style", hint: "flat vector, ukiyo-e" },
]);
function patchStyle(patch: Partial<IdeogramStyle>) {
  patchScene({ style: { ...scene.value.style, ...patch } });
}
function setStyleKind(value: string) {
  patchStyle(value === "off" ? { on: false } : { on: true, kind: value as "photo" | "art" });
}

// --- the selected box ---------------------------------------------------------------------------

const box = computed<PromptBox | null>(() =>
  selected.value.length === 1 ? (boxById.value.get(selected.value[0]!) ?? null) : null,
);
const DEFAULT_NAME = /^Box \d+$/;
function nameFrom(words: string): string {
  const name = words.replace(/["<>]/g, "").trim().split(/\s+/).slice(0, 3).join(" ");
  return name.charAt(0).toUpperCase() + name.slice(1, 32);
}
function patchBoxById(id: string, patch: Partial<PromptBox>) {
  patchScene({ boxes: scene.value.boxes.map((b) => (b.id === id ? { ...b, ...patch } : b)) });
}
function patchBox(patch: Partial<PromptBox>) {
  const current = box.value;
  if (!current) return;
  const source = (b: { kind: string; text: string; desc: string }) =>
    (b.kind === "text" && b.text.trim()) || b.desc.trim();
  const was = source(current);
  const auto = DEFAULT_NAME.test(current.name) || (!!was && current.name === nameFrom(was));
  const now = source({ ...current, ...patch });
  if (auto && now && now !== was) patch = { ...patch, name: nameFrom(now) };
  patchBoxById(current.id, patch);
}
function addBoxLora() {
  if (!box.value) return;
  patchBox({ loras: [...box.value.loras, { id: uid("o"), name: "", strength: 1, on: true }] });
}
function swapWith(other: string, close: () => void) {
  close();
  if (box.value) patchScene({ boxes: swapBoxContent(scene.value.boxes, box.value.id, other) });
}
async function openBox(id: string) {
  selected.value = [id];
  await nextTick();
  descInput.value?.$el?.querySelector?.("textarea")?.focus();
}

const inheritedRefs = computed<{ mixin: Mixin; ref: FrescoRef }[]>(() => {
  if (!box.value) return [];
  const byId = new Map(scene.value.mixins.map((m) => [m.id, m]));
  return box.value.mixins
    .map((id) => byId.get(id))
    .filter((m): m is Mixin => !!m)
    .flatMap((mixin) => mixin.refs.filter((r) => r.on).map((r) => ({ mixin, ref: r })));
});
const inheritedLoras = computed<{ mixin: Mixin; lora: FrescoLora }[]>(() => {
  if (!box.value) return [];
  const byId = new Map(scene.value.mixins.map((m) => [m.id, m]));
  return box.value.mixins
    .map((id) => byId.get(id))
    .filter((m): m is Mixin => !!m)
    .flatMap((mixin) => mixin.loras.filter((l) => l.on && l.name).map((lora) => ({ mixin, lora })));
});
const totalActiveRefs = computed(() =>
  box.value ? boxActiveRefs(scene.value, box.value).length : 0,
);
const boxRefNote = computed(() => {
  const direct = box.value?.refs.filter((r) => r.on).length ?? 0;
  const inh = inheritedRefs.value.length;
  if (!direct && !inh) return "drop or browse";
  if (inh && direct) return `${direct} box · ${inh} mixin`;
  return inh ? `${inh} from mixins` : String(direct);
});
const boxLoraNote = computed(() => {
  const direct = box.value?.loras.filter((l) => l.on && l.name).length ?? 0;
  const inh = inheritedLoras.value.length;
  if (!direct && !inh) return "";
  if (inh && direct) return `${direct} box · ${inh} mixin`;
  return inh ? `${inh} from mixins` : String(direct);
});
function shortLora(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? name;
  return base.replace(/\.[^.]+$/, "");
}

// --- multi-selection actions --------------------------------------------------------------------

function multiMixinState(ids: string[], mixinId: string): "all" | "some" | "none" {
  const chosen = scene.value.boxes.filter((b) => ids.includes(b.id));
  const count = chosen.filter((b) => b.mixins.includes(mixinId)).length;
  if (!count) return "none";
  return count === chosen.length ? "all" : "some";
}
function toggleMultiMixin(ids: string[], mixinId: string) {
  const remove = multiMixinState(ids, mixinId) === "all";
  const want = new Set(ids);
  patchScene({
    boxes: scene.value.boxes.map((b) => {
      if (!want.has(b.id)) return b;
      const mixins = remove
        ? b.mixins.filter((id) => id !== mixinId)
        : b.mixins.includes(mixinId)
          ? b.mixins
          : [...b.mixins, mixinId];
      return { ...b, mixins };
    }),
  });
}
function multiAverage(ids: string[], key: "influence" | "denoise"): number {
  const chosen = scene.value.boxes.filter((b) => ids.includes(b.id));
  if (!chosen.length) return 1;
  return +(chosen.reduce((acc, b) => acc + b[key], 0) / chosen.length).toFixed(2);
}
function patchSelectedBoxes(ids: string[], patch: Partial<PromptBox>) {
  const want = new Set(ids);
  patchScene({
    boxes: scene.value.boxes.map((b) => (want.has(b.id) ? { ...b, ...patch } : b)),
  });
}

// --- the mixins ---------------------------------------------------------------------------------

function setMixins(mixins: Mixin[]) {
  const ids = new Set(mixins.map((m) => m.id));
  patchScene({
    mixins,
    boxes: scene.value.boxes.map((b) =>
      b.mixins.every((id) => ids.has(id))
        ? b
        : { ...b, mixins: b.mixins.filter((id) => ids.has(id)) },
    ),
  });
}
function addMixin(fromBox?: string) {
  const index = scene.value.mixins.length;
  const mixin: Mixin = {
    ...readMixin({}, index),
    id: uid("c"),
    name: fromBox && !DEFAULT_NAME.test(fromBox) ? fromBox : `Mixin ${index + 1}`,
  };
  const into = fromBox !== undefined ? box.value?.id : undefined;
  patchScene({
    mixins: [...scene.value.mixins, mixin],
    boxes: into
      ? scene.value.boxes.map((b) =>
          b.id === into ? { ...b, mixins: [...b.mixins, mixin.id] } : b,
        )
      : scene.value.boxes,
  });
  editMixin(mixin.id);
}
const canPromote = computed(() => {
  const b = box.value;
  return !!b && (!!b.refs.length || !!b.loras.length || !!b.desc.trim());
});
/** Move this box's direct refs, LoRAs and description into a reusable Mixin attached to it. */
function promoteToMixin() {
  const current = box.value;
  if (!current) return;
  const index = scene.value.mixins.length;
  const mixin: Mixin = {
    ...readMixin({}, index),
    id: uid("c"),
    name: !DEFAULT_NAME.test(current.name) ? current.name : `Mixin ${index + 1}`,
    color: current.color,
    desc: current.desc,
    refs: current.refs.map((r) => ({ ...r, id: uid("r") })),
    loras: current.loras.map((l) => ({ ...l, id: uid("o") })),
  };
  patchScene({
    mixins: [...scene.value.mixins, mixin],
    boxes: scene.value.boxes.map((b) =>
      b.id === current.id
        ? { ...b, desc: "", refs: [], loras: [], mixins: [...b.mixins, mixin.id] }
        : b,
    ),
  });
  editMixin(mixin.id);
}
function editMixin(id?: string) {
  openMixin.value = id ?? scene.value.mixins[0]?.id ?? null;
  editor.value?.showTab("mixins");
}
function selectMixinBoxes(id: string) {
  selected.value = scene.value.boxes.filter((b) => b.mixins.includes(id)).map((b) => b.id);
}
function assignMixinToSelected(mixinId: string) {
  const want = new Set(selected.value);
  if (!want.size) return;
  patchScene({
    boxes: scene.value.boxes.map((b) =>
      want.has(b.id) && !b.mixins.includes(mixinId) ? { ...b, mixins: [...b.mixins, mixinId] } : b,
    ),
  });
}
function faceMixins(id: string): Mixin[] {
  const found = boxById.value.get(id);
  if (!found?.mixins.length) return [];
  return found.mixins
    .map((m) => scene.value.mixins.find((x) => x.id === m))
    .filter((m): m is Mixin => !!m);
}
function toggleMixin(id: string) {
  const current = box.value?.mixins ?? [];
  patchBox({ mixins: current.includes(id) ? current.filter((c) => c !== id) : [...current, id] });
}
const mixinNote = computed(() => {
  const named = box.value?.mixins.length ?? 0;
  return named ? `${named} here` : "";
});

// --- Mixin Library modal & chip context menu ----------------------------------------------------

const mixinLibOpen = ref(false);
const mixinSwapTarget = ref<Mixin | null>(null);
const mixinLibInitial = ref<Mixin | null>(null);

function openMixinLibrary(initial?: Mixin) {
  mixinSwapTarget.value = null;
  mixinLibInitial.value = initial ?? null;
  mixinLibOpen.value = true;
}
function openMixinSwap(target: Mixin) {
  mixinSwapTarget.value = target;
  mixinLibInitial.value = null;
  mixinLibOpen.value = true;
}
function addMixinFromLibrary(mixin: Mixin, assignSelected: boolean) {
  const want = assignSelected ? new Set(selected.value) : null;
  patchScene({
    mixins: [...scene.value.mixins, mixin],
    boxes:
      want && want.size
        ? scene.value.boxes.map((b) =>
            want.has(b.id) ? { ...b, mixins: [...b.mixins, mixin.id] } : b,
          )
        : scene.value.boxes,
  });
  openMixin.value = mixin.id;
}
function swapMixinFromLibrary(targetId: string, replacement: Mixin) {
  patchScene({
    mixins: scene.value.mixins.map((m) =>
      m.id === targetId ? { ...replacement, id: targetId } : m,
    ),
  });
  openMixin.value = targetId;
}
function syncMixinFromLibrary(libraryId: string, updated: Mixin) {
  patchScene({
    mixins: scene.value.mixins.map((m) =>
      m.libraryId === libraryId
        ? {
            ...updated,
            id: m.id,
            libraryId,
            refs: updated.refs.map((r) => ({ ...r, id: uid("r") })),
            loras: updated.loras.map((l) => ({ ...l, id: uid("o") })),
          }
        : m,
    ),
  });
}
function linkLayoutMixin(layoutMixinId: string, libraryId: string | undefined, updated?: Mixin) {
  patchScene({
    mixins: scene.value.mixins.map((m) => {
      if (m.id !== layoutMixinId) return m;
      const base = updated ? { ...updated, id: m.id } : { ...m };
      if (libraryId) base.libraryId = libraryId;
      else delete base.libraryId;
      return base;
    }),
  });
}
async function saveChipMixinToLibrary(mixin: Mixin) {
  if (!mixin.name.trim()) return;
  const clean = JSON.parse(JSON.stringify(mixin)) as Mixin;
  delete clean.libraryId;
  const saved = await saveMixinPreset({
    id: mixin.libraryId,
    name: mixin.name.trim(),
    mixin: clean,
  });
  linkLayoutMixin(mixin.id, saved.id);
}
function swapMixinBoxes(idA: string, idB: string) {
  if (idA === idB) return;
  patchScene({
    boxes: scene.value.boxes.map((b) => {
      const hasA = b.mixins.includes(idA);
      const hasB = b.mixins.includes(idB);
      if (!hasA && !hasB) return b;
      const next = b.mixins.map((id) => (id === idA ? idB : id === idB ? idA : id));
      return { ...b, mixins: [...new Set(next)] };
    }),
  });
}
function replaceBoxMixin(boxId: string, oldId: string, newId: string) {
  patchScene({
    boxes: scene.value.boxes.map((b) => {
      if (b.id !== boxId) return b;
      const next = b.mixins.map((id) => (id === oldId ? newId : id));
      return { ...b, mixins: [...new Set(next)] };
    }),
  });
}
function chipTooltip(mixin: Mixin, active: boolean): string {
  const lines = [
    active
      ? `${mixin.name} (active in this box — click to remove, right-click for options)`
      : `${mixin.name} (click to add to this box, right-click for options)`,
  ];
  if (mixin.desc.trim()) lines.push(mixin.desc.trim().slice(0, 120));
  if (mixin.trigger.trim()) lines.push(`Trigger: ${mixin.trigger.trim()}`);
  const refs = mixin.refs.filter((r) => r.on).length;
  const loras = mixin.loras.filter((l) => l.on && l.name);
  if (refs) lines.push(`${refs} reference image(s)`);
  if (loras.length) {
    lines.push(
      `LoRAs: ${loras.map((l) => `${shortLora(l.name)} (${l.strength.toFixed(2)})`).join(", ")}`,
    );
  }
  return lines.join("\n");
}

const chipMenu = ref<{ show: (e: MouseEvent) => void } | null>(null);
const chipMenuTarget = ref<Mixin | null>(null);
function openChipMenu(e: MouseEvent, mixin: Mixin) {
  chipMenuTarget.value = mixin;
  chipMenu.value?.show(e);
}
const chipMenuItems = computed<ContextMenuItem[]>(() => {
  const m = chipMenuTarget.value;
  const currentBox = box.value;
  if (!m) return [];
  const active = !!currentBox?.mixins.includes(m.id);
  const others = scene.value.mixins.filter((x) => x.id !== m.id);
  const items: ContextMenuItem[] = [
    {
      label: "Inspect in Library dialog…",
      icon: "mdi mdi-eye-outline",
      run: () => openMixinLibrary(m),
    },
    {
      label: "Edit in Mixins tab",
      icon: "mdi mdi-pencil-outline",
      run: () => editMixin(m.id),
    },
    {
      label: "Swap from Library…",
      icon: "mdi mdi-swap-horizontal",
      run: () => openMixinSwap(m),
    },
  ];
  if (currentBox && active && others.length) {
    items.push("-");
    for (const other of others.slice(0, 5)) {
      items.push({
        label: `Swap in this box → ${other.name}`,
        icon: "mdi mdi-swap-horizontal-variant",
        run: () => replaceBoxMixin(currentBox.id, m.id, other.id),
      });
    }
  }
  items.push("-");
  if (currentBox) {
    items.push({
      label: active ? "Remove from this box" : "Add to this box",
      icon: active ? "mdi mdi-minus-circle-outline" : "mdi mdi-plus-circle-outline",
      run: () => toggleMixin(m.id),
    });
  }
  items.push(
    {
      label: m.libraryId ? "Update linked Library preset" : "Save to Library & link",
      icon: m.libraryId ? "mdi mdi-cloud-upload-outline" : "mdi mdi-content-save-outline",
      disabled: !m.name.trim(),
      run: () => void saveChipMixinToLibrary(m),
    },
    "-",
    {
      label: "Delete mixin from layout",
      icon: "mdi mdi-delete-outline",
      danger: true,
      run: () => setMixins(scene.value.mixins.filter((x) => x.id !== m.id)),
    },
  );
  return items;
});

// --- prompt mode helpers ------------------------------------------------------------------------

const kindOptions = computed(() => [
  { value: "object", label: "Object", icon: "mdi mdi-cube-outline" },
  ...(scene.value.mode === "moss"
    ? [{ value: "character", label: "Character", icon: "mdi mdi-account-outline" }]
    : []),
  { value: "text", label: "Text", icon: "mdi mdi-format-text" },
]);
const citeId = computed(() =>
  scene.value.mode === "klein" && box.value ? kleinId(scene.value, box.value.id) : null,
);
const citeNote = computed(() => {
  if (!citeId.value) return scene.value.mode === "klein" ? "hidden: not in the prompt" : undefined;
  const cited = scene.value.prompt.includes(`<${citeId.value}>`);
  return `<${citeId.value}>${cited ? "" : " — not cited yet"}`;
});
const descPlaceholder = computed(() =>
  scene.value.mode === "klein"
    ? "Opens with what it is: shape, material, colours, markings — no position"
    : "What belongs in this box (empty = uses its mixins / scene prompt)",
);
function cite(id: string) {
  const prompt = scene.value.prompt.trimEnd();
  patchScene({ prompt: `${prompt}${prompt ? " " : ""}<${id}>` });
}

// --- painted shapes: grown mask preview & shapeFor clipping -------------------------------------

const grown = ref<Record<string, string>>({});
function growFractions(b: PromptBox): [number, number] {
  return [
    b.grow / Math.max(1, b.w * scene.value.width),
    b.grow / Math.max(1, b.h * scene.value.height),
  ];
}
function growKey(b: PromptBox): string | null {
  if (!b.mask || !b.grow) return null;
  const [fx, fy] = growFractions(b);
  return `${fx.toFixed(4)},${fy.toFixed(4)},${b.mask}`;
}
function maskView(id: string): string | null {
  const b = boxById.value.get(id);
  const key = b && growKey(b);
  return key ? (grown.value[key] ?? null) : null;
}
function maskClips(id: string): boolean {
  return boxById.value.get(id)?.shapeFor !== "refs";
}
watchEffect(() => {
  for (const b of scene.value.boxes) {
    const key = growKey(b);
    if (!key || grown.value[key]) continue;
    const [fx, fy] = growFractions(b);
    void growMask(b.mask!, fx, fy).then((url) => {
      grown.value = { ...grown.value, [key]: url };
    });
  }
});

// --- dropping media on the canvas ---------------------------------------------------------------

function boxAround(x: number, y: number): string | undefined {
  const w = 0.3;
  const h = Math.min(0.9, (w * scene.value.width) / scene.value.height);
  return editor.value?.addBox({
    x: Math.min(1 - w, Math.max(0, x - w / 2)),
    y: Math.min(1 - h, Math.max(0, y - h / 2)),
    w,
    h,
  });
}

async function onDrop(e: DragEvent, at: { x: number; y: number }, hit: string | null) {
  const media = (await readMediaDrop(e, "fresco")).filter(isImage);
  if (!media.length) return;
  const refs = media.map((m) => newRef(refFromMedia(m)));
  const target = hit ? boxById.value.get(hit) : undefined;
  if (target) {
    patchBoxById(target.id, { refs: [...target.refs, ...refs] });
    selected.value = [target.id];
    return;
  }
  const id = boxAround(at.x, at.y);
  if (id) patchBoxById(id, { refs });
}

// --- SAM3: wand, click points, and segmenting on Layout -----------------------------------------

const wand = ref(false);
function setWand(on: boolean) {
  wand.value = on;
  if (on) tool.value = "select";
}

function stageFraction(e: PointerEvent): { x: number; y: number } | null {
  const rect = (e.currentTarget as HTMLElement | null)?.closest(".zbc")?.getBoundingClientRect();
  if (!rect?.width) return null;
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  return {
    x: clamp((e.clientX - rect.left) / rect.width),
    y: clamp((e.clientY - rect.top) / rect.height),
  };
}

function onWandClick(e: PointerEvent) {
  const at = stageFraction(e);
  if (!at) return;
  const { x, y } = at;
  const point = { x, y, include: !e.altKey };
  const inside = (b: PromptBox) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h;
  const current = box.value;
  const visible = scene.value.boxes.filter((b) => !b.hidden);
  const under = boxAt(visible, x, y);
  const hit = current && !current.hidden && inside(current) ? current : under;
  if (hit) {
    selected.value = [hit.id];
    patchBoxById(hit.id, { points: [...(boxById.value.get(hit.id)?.points ?? []), point] });
  } else {
    if (!point.include) return;
    const id = boxAround(x, y);
    if (id) patchBoxById(id, { points: [point] });
  }
  scheduleSegment();
}

const pointDrag = ref<{
  index: number;
  x: number;
  y: number;
  startX: number;
  startY: number;
  moved: boolean;
} | null>(null);
function pointStyle(pt: FrescoPoint, i: number) {
  const at = pointDrag.value?.index === i ? pointDrag.value : pt;
  return { left: `${at.x * 100}%`, top: `${at.y * 100}%` };
}
function onPointDown(e: PointerEvent, index: number) {
  const pt = box.value?.points[index];
  if (!wand.value || !pt) return;
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  pointDrag.value = { index, x: pt.x, y: pt.y, startX: e.clientX, startY: e.clientY, moved: false };
}
function onPointMove(e: PointerEvent) {
  const drag = pointDrag.value;
  const at = drag && stageFraction(e);
  if (!drag || !at) return;
  const moved = drag.moved || Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 3;
  pointDrag.value = { ...drag, ...(moved ? at : {}), moved };
}
function onPointUp(e: PointerEvent, index: number) {
  const drag = pointDrag.value;
  pointDrag.value = null;
  const current = box.value;
  if (!drag || drag.index !== index || !current) return;
  const points = current.points.flatMap((pt, i) => {
    if (i !== index) return [pt];
    if (drag.moved) return [{ ...pt, x: drag.x, y: drag.y }];
    if (e.altKey) return [{ ...pt, include: !pt.include }];
    return [];
  });
  patchBox({ points });
  if (points.length) scheduleSegment();
}

const segmenting = ref(false);
const segmentError = ref("");
const captureNote = computed(() => {
  if (segmenting.value) return "segmenting…";
  const n = box.value?.points.length ?? 0;
  return n ? `${n} point${n === 1 ? "" : "s"} · source photo` : "masks wired source photo";
});
let segmentTimer: ReturnType<typeof setTimeout> | undefined;
let segmentWatchdog: ReturnType<typeof setTimeout> | undefined;
function finishSegment(error = "") {
  segmenting.value = false;
  clearTimeout(segmentWatchdog);
  segmentWatchdog = undefined;
  segmentError.value = error;
}
function scheduleSegment() {
  clearTimeout(segmentTimer);
  segmentError.value = "";
  segmentTimer = setTimeout(() => void segment(), 700);
}
async function segment() {
  clearTimeout(segmentTimer);
  if (segmenting.value || props.node?.id == null) return;
  segmenting.value = true;
  segmentError.value = "";
  clearTimeout(segmentWatchdog);
  segmentWatchdog = setTimeout(() => finishSegment("Segmentation timed out."), 60_000);
  try {
    await app.queuePrompt(0, 1, [String(props.node.id)]);
  } catch (e) {
    finishSegment((e as Error)?.message || "Could not queue segmentation.");
  }
}
const apiEvents = api as unknown as {
  addEventListener(t: string, f: (e: Event) => void): void;
  removeEventListener(t: string, f: (e: Event) => void): void;
};
function onExecutionError(e: Event) {
  if (!segmenting.value) return;
  const detail = (e as CustomEvent<{ exception_message?: string }>).detail;
  finishSegment(detail?.exception_message?.trim() || "Segmentation failed.");
}
function onExecutionInterrupted() {
  if (!segmenting.value) return;
  finishSegment("Segmentation interrupted.");
}
function onExecutionSuccess() {
  if (!segmenting.value) return;
  finishSegment("");
}
onMounted(() => {
  apiEvents.addEventListener("execution_error", onExecutionError);
  apiEvents.addEventListener("execution_interrupted", onExecutionInterrupted);
  apiEvents.addEventListener("execution_success", onExecutionSuccess);
});
watch(
  () => box.value?.id,
  () => {
    segmentError.value = "";
  },
);
function takeMasks(masks: { id: string; mask: string; source: string }[]) {
  const byId = new Map(masks.map((m) => [m.id, m]));
  if (byId.size)
    patchScene({
      boxes: scene.value.boxes.map((b) => {
        const m = byId.get(b.id);
        return m ? { ...b, mask: m.mask, maskSource: m.source } : b;
      }),
    });
}
function onKey(e: KeyboardEvent) {
  if (!(e.target as HTMLElement | null)?.closest?.(".le-stage")) return;
  if ((e.key === "w" || e.key === "W") && !e.ctrlKey && !e.metaKey) {
    setWand(!wand.value);
    e.preventDefault();
    e.stopPropagation();
  }
}

// --- the agent, and the photo + masks a run sends back ------------------------------------------

function applyAgent(result: AgentLayout, how: "replace" | "add") {
  const kept = how === "add" ? scene.value.boxes : [];
  const boxes = [...kept];
  for (const b of result.boxes) {
    boxes.push(
      readBox(
        { ...b, id: boxId(), name: b.name || `Box ${boxes.length + 1}`, color: nextColor(boxes) },
        boxes.length,
      ),
    );
  }
  patchScene({
    boxes,
    ...(how === "replace" || !scene.value.prompt.trim() ? { prompt: result.prompt } : {}),
    ...(result.background && (how === "replace" || !scene.value.background.trim())
      ? { background: result.background }
      : {}),
  });
  selected.value = [];
}

useSentBack(
  "layout",
  () => String(props.node?.id ?? ""),
  (outputData, image) => {
    if (Array.isArray(outputData.layout_masks)) {
      takeMasks(outputData.layout_masks as { id: string; mask: string; source: string }[]);
    }
    const err = Array.isArray(outputData.layout_segment_error)
      ? String(outputData.layout_segment_error[0] ?? "")
      : "";
    if (segmenting.value || err) finishSegment(err);
    if (image) editor.value?.offerBackdrop(image);
  },
);

// --- the prompt out -----------------------------------------------------------------------------

const output = ref("");
const warnings = ref<string[]>([]);
const compiling = ref(false);
let timer: ReturnType<typeof setTimeout> | undefined;
let ticket = 0;
async function recompile() {
  const mine = ++ticket;
  compiling.value = true;
  try {
    const result = await compileLayout(JSON.parse(JSON.stringify(scene.value)));
    if (mine !== ticket) return;
    output.value = result.prompt;
    warnings.value = result.warnings;
  } catch (e) {
    if (mine === ticket) warnings.value = [`Couldn't write the prompt: ${(e as Error).message}`];
  } finally {
    if (mine === ticket) compiling.value = false;
  }
}
watch(
  scene,
  () => {
    clearTimeout(timer);
    timer = setTimeout(recompile, 250);
  },
  { deep: true, immediate: true },
);
onBeforeUnmount(() => {
  clearTimeout(timer);
  clearTimeout(segmentTimer);
  clearTimeout(segmentWatchdog);
  apiEvents.removeEventListener("execution_error", onExecutionError);
  apiEvents.removeEventListener("execution_interrupted", onExecutionInterrupted);
  apiEvents.removeEventListener("execution_success", onExecutionSuccess);
});
const copied = ref(false);
async function copyOutput() {
  try {
    await navigator.clipboard.writeText(output.value);
    copied.value = true;
    setTimeout(() => (copied.value = false), 1500);
  } catch {
    copied.value = false;
  }
}
</script>

<style scoped>
.pl-cog {
  margin-left: 2px;
  font-size: 13px;
  opacity: 0.7;
}
.pl-host {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.pl-choose {
  position: absolute;
  inset: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-bg, #141418) 88%, transparent);
  backdrop-filter: blur(2px);
}
.pl-choose-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: 720px;
  color: var(--zen-text, #e5e5ea);
}
.pl-choose-title {
  font-size: 16px;
  font-weight: 600;
}
.pl-choose-note {
  font-size: 12px;
  color: var(--zen-muted, #9aa0aa);
}
.pl-note {
  margin: 0;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 60%, transparent);
  font-size: 11px;
  line-height: 1.45;
  color: var(--zen-muted, #9aa0aa);
}
.pl-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pl-seg {
  display: flex;
  width: 100%;
}
.pl-seg :deep(.zen-tg-b) {
  flex: 1 1 0;
  min-width: 0;
}
.pl-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.pl-face {
  position: absolute;
  top: 4px;
  right: 4px;
  display: flex;
  gap: 2px;
  pointer-events: none;
}
.pl-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 24px;
  padding: 0 9px 0 3px;
  border: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 80%, transparent);
  border-radius: 11px;
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
  cursor: pointer;
}
.pl-chip:hover {
  color: var(--zen-text, #e5e5ea);
}
.pl-chip.on {
  border-color: var(--zen-accent, #3b82f6);
  background: color-mix(in srgb, var(--zen-accent, #3b82f6) 22%, transparent);
  color: var(--zen-text, #e5e5ea);
}
.pl-chip.partial {
  border-style: dashed;
  border-color: var(--zen-accent, #3b82f6);
  color: var(--zen-text, #e5e5ea);
}
.pl-inherited {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 4px;
}
.pl-inh-ref,
.pl-inh-lora {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  max-width: 100%;
  padding: 2px 6px 2px 3px;
  border: 1px dashed color-mix(in srgb, var(--zen-border, #34343c) 90%, transparent);
  border-radius: 5px;
  background: color-mix(in srgb, var(--zen-text, #fff) 3%, transparent);
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 10.5px;
  cursor: pointer;
}
.pl-inh-ref:hover,
.pl-inh-lora:hover {
  border-color: var(--zen-accent, #3b82f6);
  color: var(--zen-text, #e5e5ea);
}
.pl-inh-ref img {
  width: 18px;
  height: 18px;
  border-radius: 3px;
  object-fit: cover;
}
.pl-inh-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 120px;
}
.pl-inh-str {
  font-variant-numeric: tabular-nums;
  opacity: 0.75;
}
.pl-hint {
  margin: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.pl-seg-err {
  display: flex;
  align-items: flex-start;
  gap: 5px;
  margin: 0;
  padding: 5px 7px;
  border-radius: var(--zen-radius, 6px);
  background: color-mix(in srgb, var(--zen-warning, #f5b041) 14%, transparent);
  color: var(--zen-warning, #f5b041);
  font-size: 11px;
  line-height: 1.35;
}
.pl-grid {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
}
.pl-blend-row {
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr) 82px;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
}
.pl-pairs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 6px 10px;
}
.pl-pair {
  display: grid;
  grid-template-columns: 52px minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
}
.pl-row-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.pl-sub {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.pl-menu {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
}
.pl-menu-title {
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.pl-menu .zen-btn {
  justify-content: flex-start;
}
.pl-swatch {
  flex: none;
  width: 12px;
  height: 12px;
  border-radius: 3px;
}
.pl-wand {
  position: absolute;
  inset: 0;
  cursor: crosshair;
}
.pl-point {
  position: absolute;
  width: 10px;
  height: 10px;
  margin: -5px 0 0 -5px;
  border: 2px solid #fff;
  border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 0 1px rgb(0 0 0 / 50%);
  pointer-events: none;
}
.pl-point.live {
  pointer-events: auto;
  cursor: grab;
  touch-action: none;
}
.pl-point.live:hover {
  transform: scale(1.3);
}
.pl-point.out {
  background: #ef4444;
}
.pl-status {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 24px;
  padding: 0 8px;
  border: 0;
  border-radius: var(--zen-radius, 7px);
  background: none;
  color: var(--zen-success, #4ade80);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.pl-status:hover {
  background: color-mix(in srgb, var(--zen-text, #e5e5ea) 8%, transparent);
}
.pl-status.warn {
  color: var(--zen-warning, #f5b041);
}
.pl-count {
  color: var(--zen-muted, #9aa0aa);
  font-variant-numeric: tabular-nums;
}
.pl-pop {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: min(340px, 88vw);
  box-sizing: border-box;
  padding: 10px;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.pl-pop-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.pl-pop-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.pl-out {
  margin: 0;
  max-height: 320px;
  overflow: auto;
  padding: 6px 8px;
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface-2, #26262d);
  font-family: var(--zen-mono, ui-monospace, monospace);
  font-size: 10.5px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  user-select: text;
}
.pl-warn {
  margin: 0;
  padding-left: 16px;
  font-size: 11px;
  color: var(--zen-warning, #f5b041);
}
</style>
