// Component styles plus the bridge that maps ComfyUI's CSS vars onto --zen-* tokens, so the
// components theme correctly without the ZenKit runtime.
import "@nynxz/zenkit-ui/style.css";
import "@nynxz/zenkit-ui/comfy-bridge.css";

import { DISPLAY_NAME, discoverWidgets, registerNodes, widgetTypes } from "@/framework";

// Every .vue in frontend/widgets/ is a node widget; the filename determines its io type.
const widgets = import.meta.glob("./widgets/*.vue", { eager: true });

registerNodes(discoverWidgets(widgets));

console.log(`[${DISPLAY_NAME}] widgets: ${widgetTypes(widgets).join(", ") || "(none)"}`);
