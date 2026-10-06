// User settings for zenkit-ui's LoRA picker and details view, applied to its shared store.

import { app } from "@comfy/app";
import { loraLibrary } from "@nynxz/zenkit-ui";

import { nodeId, settingId } from "@/framework";

app.registerExtension({
  name: `${nodeId("Lora")}.Settings`,
  settings: [
    {
      id: settingId("Lora.FolderNav"),
      name: "Browse LoRAs by folder",
      category: ["Nynxz", "LoRA picker", "Folders"],
      type: "boolean",
      defaultValue: true,
      tooltip:
        "Step through LoRA folders in the picker dropdown. Off lists every LoRA in one flat " +
        "list. Typing always searches every folder.",
      onChange: (on: unknown) => loraLibrary.setFolderNav(on !== false),
    },
    {
      id: settingId("Lora.BlurMature"),
      name: "Blur mature example images",
      category: ["Nynxz", "LoRA picker", "Mature"],
      type: "boolean",
      defaultValue: true,
      tooltip: "Blur LoRA examples Civitai rates R or above until revealed. Off always shows them.",
      onChange: (on: unknown) => loraLibrary.setBlurMature(on !== false),
    },
  ],
} as never);
