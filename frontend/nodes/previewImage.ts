// Preview Image reuses Save Image's SavePreview widget. Widget discovery binds one node def
// per widget FILE, so a second node sharing that io type declares itself here instead; the
// widget options are imported rather than restated so the two can't drift.

import { defineNode, typeId } from "@/framework";
import { previewRecords } from "@/lib/saveApi";
import SavePreview, { widgetOptions } from "@/widgets/SavePreview.vue";

export default defineNode({
  is: "nynxz.Preview.Image",
  minSize: [300, 320],
  hideOutputImages: true, // only our preview, never ComfyUI's default strip
  widgets: [{ ...widgetOptions, type: typeId("SavePreview"), component: SavePreview }],
  // Preview Image writes to temp under its own ui key (see nodes/preview/preview_image.py).
  output: { widget: widgetOptions.name!, from: previewRecords("preview", "temp") },
});
