// Suggestions for the style fields of an Ideogram caption — phrases Ideogram handles well (from
// IdeogramHelper; looks it handles poorly, such as pixel art and fisheye, are left out).

export const STYLE_PRESETS: Record<
  "photo" | "lighting" | "aesthetics" | "medium" | "art_style",
  string[]
> = {
  photo: [
    "35mm, f/1.8, shallow depth of field",
    "50mm, f/1.4, natural perspective",
    "85mm portrait, f/2, creamy bokeh",
    "24mm wide-angle, deep focus",
    "100mm macro, extreme close-up",
    "200mm telephoto, compressed background",
    "full-frame DSLR, crisp detail",
    "phone camera, casual snapshot",
    "cinematic, anamorphic widescreen",
  ],
  lighting: [
    "golden hour, warm side light",
    "blue hour, cool ambient",
    "soft diffused daylight",
    "hard midday sun, strong shadows",
    "rim light, backlit silhouette",
    "studio softbox, even lighting",
    "dramatic chiaroscuro, single source",
    "neon, colourful city glow",
    "candlelight, warm flicker",
    "overcast, flat soft light",
    "low-key, moody darkness",
    "high-key, bright and airy",
  ],
  aesthetics: [
    "moody, cinematic, desaturated",
    "vibrant, high contrast, punchy",
    "soft, dreamy, pastel",
    "gritty, raw, documentary",
    "clean, minimal, modern",
    "vintage film, grainy, faded",
    "epic, dramatic, grand scale",
    "cozy, warm, intimate",
    "ethereal, glowing, surreal",
  ],
  medium: [
    "photograph",
    "illustration",
    "3d_render",
    "painting",
    "digital_art",
    "watercolor",
    "oil_painting",
    "anime",
    "concept_art",
  ],
  art_style: [
    "watercolour painting, soft washes",
    "oil painting, visible brushstrokes",
    "anime, cel-shaded",
    "comic book, bold ink",
    "pencil sketch, cross-hatching",
    "concept art, painterly",
  ],
};
