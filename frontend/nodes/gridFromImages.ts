// Grid From Images renders the grid and previews the result in the SHARED save_preview
// widget (registered by Save Image's def), so it declares no widget of its own — which is
// why it lives here rather than beside a .vue.

import { defineNode } from "@/framework";
import { previewRecords } from "@/lib/saveApi";

export default defineNode({
  is: "nynxz.Preview.GridFromImages",
  minSize: [300, 320],
  hideOutputImages: true,
  output: { widget: "save_preview", from: previewRecords("preview", "temp") },
  slotLinks: [
    { input: "style", spawn: "nynxz.Preview.GridStyle" },
    { input: "background", spawn: "nynxz.Preview.Background" },
  ],
});
