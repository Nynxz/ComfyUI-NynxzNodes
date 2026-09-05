import { zenPluginConfig } from "@nynxz/zenkit-nodekit/vite";

// The whole build comes from the shared preset: @comfy/* externalised and rewritten,
// CSS folded into the single main.js ComfyUI loads, and NODE_ENV defined for Vue.
export default zenPluginConfig({
  name: "comfyui-nynxznodes",
  configUrl: import.meta.url,
  srcDir: "./frontend",
  outDir: "web", // matches WEB_DIRECTORY in __init__.py
});
