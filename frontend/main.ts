// Component styles plus the bridge that maps ComfyUI's CSS vars onto --zen-* tokens, so the
// components theme correctly without the ZenKit runtime.
import "@nynxz/zenkit-ui/style.css";
import "@nynxz/zenkit-ui/comfy-bridge.css";

// zenkit-ui owns the LoRA picker UI; this pack owns the routes behind it.
import { setLoraDetailOpener, setLoraSource } from "@nynxz/zenkit-ui";

import { loraSource, openLoraDetailPanel } from "@/lib/loraApi";

import {
  DISPLAY_NAME,
  discoverNodes,
  discoverWidgets,
  registerNodes,
  widgetTypes,
} from "@/framework";

// Every .vue in frontend/widgets/ is a node widget; the filename determines its io type.
const widgets = import.meta.glob("./widgets/*.vue", { eager: true });
// Extra node defs for nodes that REUSE a widget type already claimed by a widget file
// (discovery binds one node def per widget file). See frontend/nodes/previewImage.ts.
const extraNodes = import.meta.glob("./nodes/*.ts", { eager: true });

setLoraSource(loraSource);
setLoraDetailOpener(openLoraDetailPanel);

registerNodes([...discoverWidgets(widgets), ...discoverNodes(extraNodes)]);

// Frontend-only nodes (no Python class) register themselves.
import "@/groupSwitcher/register";
// LoRA Loader's "apply to" setting reshapes its sockets.
import "@/lib/loadImageOutputs";
import "@/lib/loraMode";
// LoRA picker user settings (folder browsing).
import "@/lib/loraSettings";

console.log(`[${DISPLAY_NAME}] widgets: ${widgetTypes(widgets).join(", ") || "(none)"}`);
