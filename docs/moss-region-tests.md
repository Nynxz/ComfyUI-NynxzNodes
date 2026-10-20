# MOSS-Image region tests

Seven layouts for stress-testing MOSS-Image's layout prompts, with and without Regions Apply.
Paste each **Layout JSON** into a Layout node in MOSS mode: select the canvas and paste, or use the
editor's JSON paste.

## Setup

Use one graph, run three ways. Keep the seed, sampler, steps and CFG identical across a case's runs,
and do every run on three seeds (1, 2, 3); one good seed proves nothing.

| Run | Wiring |
|---|---|
| **R0** layout only | Layout → Layout Extras `prompt` → text encoder → positive. The model goes straight to the sampler. |
| **R1** + regions | As R0, plus Layout Extras `regions` and the CLIP into **Regions Apply**, `isolation` 0. Its model goes to the sampler. |
| **R2** + isolation | As R1 with `isolation` 0.5. |

```
Layout (MOSS) ──> Layout Extras ─ prompt ──> CLIP Text Encode ──> positive
                                └ regions ─┐
model ────────────────────────────────────> Regions Apply (+ clip) ──> KSampler   (R1, R2)
```

Score each image per case on its checklist: **pass**, **partial** or **fail** for each point, and
note anything the checklist doesn't cover.

Cases A-E each isolate one failure. F is the hard one, an eight-panel anime storyboard page with
recurring characters, and G is the same page written the way the reported layout was. Start with
F and G if you only have time for one pair.

**Two rules the cases follow, which also apply to real prompts:**

- **Keep the scene caption about style, not content.** With Regions Apply on, every box reads the
  caption plus its own description. A caption that says "four panels" or names the story makes
  every panel try to draw all of it. Say "four panels" with MOSS's canvas setting (`multi_panel`),
  and put the story in the boxes.
- **Give each box what makes it different.** Boxes that repeat the same words leave the model only
  their coordinates to tell them apart (case A). Repeat a character's identity verbatim (it keeps
  them consistent), but lead with what changes: shot, place, action, expression (case B).

## Case A — Same character, same words (the reported failure)

Four panels whose boxes all describe Amber in almost the same words. This is the reported failure case: the panels have nothing but their numbers to tell them apart.

**Look for:** Which panels show Amber at all; whether any panel is empty, duplicated, or holds two Ambers; whether her coat and hair stay the same in every panel.

| Seed | R0 | R1 | R2 | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

<details><summary>Layout JSON (paste into the Layout node)</summary>

```json
{
  "nynxzLayout": 1,
  "mode": "moss",
  "layout": {
    "mode": "moss",
    "width": 1024,
    "height": 1024,
    "prompt": "Comic book art, clean ink lines, flat colours, white gutters.",
    "meta": [],
    "canvas": "multi_panel",
    "palette": [],
    "boxes": [
      {
        "id": "b1",
        "name": "Panel 1",
        "x": 0.02,
        "y": 0.02,
        "w": 0.46,
        "h": 0.46,
        "kind": "object",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain",
        "text": "",
        "palette": []
      },
      {
        "id": "b2",
        "name": "Panel 2",
        "x": 0.52,
        "y": 0.02,
        "w": 0.46,
        "h": 0.46,
        "kind": "object",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain, looking up",
        "text": "",
        "palette": []
      },
      {
        "id": "b3",
        "name": "Panel 3",
        "x": 0.02,
        "y": 0.52,
        "w": 0.46,
        "h": 0.46,
        "kind": "object",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain",
        "text": "",
        "palette": []
      },
      {
        "id": "b4",
        "name": "Panel 4",
        "x": 0.52,
        "y": 0.52,
        "w": 0.46,
        "h": 0.46,
        "kind": "object",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain, looking up",
        "text": "",
        "palette": []
      }
    ]
  }
}
```

</details>

<details><summary>The MOSS prompt it compiles to (R0's positive)</summary>

```json
{
  "summary": {
    "canvas": "multi_panel",
    "caption": "Comic book art, clean ink lines, flat colours, white gutters."
  },
  "items": [
    {
      "label": "object",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain",
      "bbox": [
        -492,
        -492,
        -20,
        -20
      ]
    },
    {
      "label": "object",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain, looking up",
      "bbox": [
        -492,
        20,
        -20,
        492
      ]
    },
    {
      "label": "object",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain",
      "bbox": [
        20,
        -492,
        492,
        -20
      ]
    },
    {
      "label": "object",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain, looking up",
      "bbox": [
        20,
        20,
        492,
        492
      ]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

</details>

What a box reads under Regions Apply, for example **Panel 4**:

> Comic book art, clean ink lines, flat colours, white gutters. Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater, standing in the rain, looking up.

## Case B — Same character, different beats

The same four panels written the way that should work: each panel box says what is different (shot, place, action), and Amber gets her own box inside each panel with her description word for word plus that panel's pose and expression. Eight boxes; the character boxes sit on top, so they own their area and the panel owns the rest.

**Look for:** Each panel shows its own beat (bed, kitchen, street, doorway) and not another's; Amber appears once per panel, in her box, with that panel's pose; the same face, hair and coat in all four.

| Seed | R0 | R1 | R2 | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

<details><summary>Layout JSON (paste into the Layout node)</summary>

```json
{
  "nynxzLayout": 1,
  "mode": "moss",
  "layout": {
    "mode": "moss",
    "width": 1024,
    "height": 1024,
    "prompt": "Comic book art, clean ink lines, flat colours, white gutters.",
    "meta": [],
    "canvas": "multi_panel",
    "palette": [],
    "boxes": [
      {
        "id": "b1",
        "name": "Panel 1",
        "x": 0.02,
        "y": 0.02,
        "w": 0.46,
        "h": 0.46,
        "kind": "object",
        "desc": "Panel 1, wide shot at dawn: a small bedroom, rain on the window, an alarm clock on the nightstand.",
        "text": "",
        "palette": []
      },
      {
        "id": "b11",
        "name": "Amber 1",
        "x": 0.135,
        "y": 0.158,
        "w": 0.253,
        "h": 0.2852,
        "kind": "character",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She sits up in bed, hair messy, rubbing her eyes, sleepy.",
        "text": "",
        "palette": []
      },
      {
        "id": "b2",
        "name": "Panel 2",
        "x": 0.52,
        "y": 0.02,
        "w": 0.46,
        "h": 0.46,
        "kind": "object",
        "desc": "Panel 2, medium shot: a cramped kitchen, a steaming kettle, a mug with a cat on it.",
        "text": "",
        "palette": []
      },
      {
        "id": "b12",
        "name": "Amber 2",
        "x": 0.681,
        "y": 0.0752,
        "w": 0.23,
        "h": 0.391,
        "kind": "character",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She pours coffee, half awake, yawning.",
        "text": "",
        "palette": []
      },
      {
        "id": "b3",
        "name": "Panel 3",
        "x": 0.02,
        "y": 0.52,
        "w": 0.46,
        "h": 0.46,
        "kind": "object",
        "desc": "Panel 3, wide shot: a busy city street in heavy rain, a red bus passing, puddles.",
        "text": "",
        "palette": []
      },
      {
        "id": "b13",
        "name": "Amber 3",
        "x": 0.043,
        "y": 0.635,
        "w": 0.184,
        "h": 0.3312,
        "kind": "character",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She runs with her hood up, splashing through a puddle, determined.",
        "text": "",
        "palette": []
      },
      {
        "id": "b4",
        "name": "Panel 4",
        "x": 0.52,
        "y": 0.52,
        "w": 0.46,
        "h": 0.46,
        "kind": "object",
        "desc": "Panel 4, close-up: an office doorway, warm light inside.",
        "text": "",
        "palette": []
      },
      {
        "id": "b14",
        "name": "Amber 4",
        "x": 0.589,
        "y": 0.5568,
        "w": 0.322,
        "h": 0.414,
        "kind": "character",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. Close-up of her face, soaking wet, grinning in triumph, giving a thumbs up.",
        "text": "",
        "palette": []
      }
    ]
  }
}
```

</details>

<details><summary>The MOSS prompt it compiles to (R0's positive)</summary>

```json
{
  "summary": {
    "canvas": "multi_panel",
    "caption": "Comic book art, clean ink lines, flat colours, white gutters."
  },
  "items": [
    {
      "label": "object",
      "caption": "Panel 1, wide shot at dawn: a small bedroom, rain on the window, an alarm clock on the nightstand.",
      "bbox": [
        -492,
        -492,
        -20,
        -20
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She sits up in bed, hair messy, rubbing her eyes, sleepy.",
      "bbox": [
        -351,
        -374,
        -58,
        -114
      ]
    },
    {
      "label": "object",
      "caption": "Panel 2, medium shot: a cramped kitchen, a steaming kettle, a mug with a cat on it.",
      "bbox": [
        -492,
        20,
        -20,
        492
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She pours coffee, half awake, yawning.",
      "bbox": [
        -435,
        185,
        -34,
        421
      ]
    },
    {
      "label": "object",
      "caption": "Panel 3, wide shot: a busy city street in heavy rain, a red bus passing, puddles.",
      "bbox": [
        20,
        -492,
        492,
        -20
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She runs with her hood up, splashing through a puddle, determined.",
      "bbox": [
        138,
        -468,
        478,
        -279
      ]
    },
    {
      "label": "object",
      "caption": "Panel 4, close-up: an office doorway, warm light inside.",
      "bbox": [
        20,
        20,
        492,
        492
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. Close-up of her face, soaking wet, grinning in triumph, giving a thumbs up.",
      "bbox": [
        58,
        91,
        483,
        421
      ]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

</details>

What a box reads under Regions Apply, for example **Amber 4**:

> Comic book art, clean ink lines, flat colours, white gutters. Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. Close-up of her face, soaking wet, grinning in triumph, giving a thumbs up.

## Case C — Two characters, swapped attributes

Attribute binding: two people side by side with deliberately contrasting hair, clothes and props. Failure shows as swapped colours or a merged person.

**Look for:** Amber has the copper bob, yellow coat and red umbrella; Kai has the black hair, denim jacket and green umbrella. Any swap counts as a fail, and so does one person with mixed features.

| Seed | R0 | R1 | R2 | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

<details><summary>Layout JSON (paste into the Layout node)</summary>

```json
{
  "nynxzLayout": 1,
  "mode": "moss",
  "layout": {
    "mode": "moss",
    "width": 1344,
    "height": 768,
    "prompt": "A street corner on a rainy evening, cinematic photo, shallow depth of field, neon reflections.",
    "meta": [],
    "canvas": "",
    "palette": [],
    "boxes": [
      {
        "id": "b1",
        "name": "Amber",
        "x": 0.06,
        "y": 0.18,
        "w": 0.4,
        "h": 0.8,
        "kind": "character",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She holds a closed red umbrella and laughs.",
        "text": "",
        "palette": []
      },
      {
        "id": "b2",
        "name": "Kai",
        "x": 0.54,
        "y": 0.14,
        "w": 0.4,
        "h": 0.84,
        "kind": "character",
        "desc": "Kai, a tall young man with straight black hair to his chin, brown eyes and a thin scar over his left eyebrow, wearing a blue denim jacket over a white hoodie. He holds an open green umbrella and frowns.",
        "text": "",
        "palette": []
      }
    ]
  }
}
```

</details>

<details><summary>The MOSS prompt it compiles to (R0's positive)</summary>

```json
{
  "summary": {
    "caption": "A street corner on a rainy evening, cinematic photo, shallow depth of field, neon reflections."
  },
  "items": [
    {
      "label": "character",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She holds a closed red umbrella and laughs.",
      "bbox": [
        -246,
        -592,
        369,
        -53
      ]
    },
    {
      "label": "character",
      "caption": "Kai, a tall young man with straight black hair to his chin, brown eyes and a thin scar over his left eyebrow, wearing a blue denim jacket over a white hoodie. He holds an open green umbrella and frowns.",
      "bbox": [
        -277,
        53,
        369,
        592
      ]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

</details>

What a box reads under Regions Apply, for example **Kai**:

> A street corner on a rainy evening, cinematic photo, shallow depth of field, neon reflections. Kai, a tall young man with straight black hair to his chin, brown eyes and a thin scar over his left eyebrow, wearing a blue denim jacket over a white hoodie. He holds an open green umbrella and frowns.

## Case D — Small things and text

Many small boxes, each with one distinct object, plus a text box. Failure shows as objects in the wrong box, missing objects, or a misspelled sign.

**Look for:** Each object is in its own box and none is missing; the sign reads exactly OPEN 24H in pink neon; the cat sits on the bin.

| Seed | R0 | R1 | R2 | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

<details><summary>Layout JSON (paste into the Layout node)</summary>

```json
{
  "nynxzLayout": 1,
  "mode": "moss",
  "layout": {
    "mode": "moss",
    "width": 1024,
    "height": 1024,
    "prompt": "A narrow back alley at night, photo, wet cobblestones, a single street lamp.",
    "meta": [],
    "canvas": "",
    "palette": [],
    "boxes": [
      {
        "id": "b1",
        "name": "Door",
        "x": 0.05,
        "y": 0.25,
        "w": 0.22,
        "h": 0.6,
        "kind": "object",
        "desc": "a heavy green metal door with a rusty handle",
        "text": "",
        "palette": []
      },
      {
        "id": "b2",
        "name": "Sign",
        "x": 0.32,
        "y": 0.08,
        "w": 0.36,
        "h": 0.14,
        "kind": "text",
        "desc": "a pink neon sign",
        "text": "OPEN 24H",
        "palette": [
          "#FF4FA3"
        ]
      },
      {
        "id": "b3",
        "name": "Bicycle",
        "x": 0.33,
        "y": 0.55,
        "w": 0.3,
        "h": 0.32,
        "kind": "object",
        "desc": "a red vintage bicycle leaning on the wall",
        "text": "",
        "palette": []
      },
      {
        "id": "b4",
        "name": "Bin",
        "x": 0.7,
        "y": 0.72,
        "w": 0.22,
        "h": 0.22,
        "kind": "object",
        "desc": "a dented silver metal bin",
        "text": "",
        "palette": []
      },
      {
        "id": "b5",
        "name": "Cat",
        "x": 0.72,
        "y": 0.58,
        "w": 0.16,
        "h": 0.16,
        "kind": "object",
        "desc": "a black cat sitting on a metal bin",
        "text": "",
        "palette": []
      },
      {
        "id": "b6",
        "name": "Lamp",
        "x": 0.78,
        "y": 0.04,
        "w": 0.14,
        "h": 0.3,
        "kind": "object",
        "desc": "an old cast-iron street lamp, glowing orange",
        "text": "",
        "palette": []
      }
    ]
  }
}
```

</details>

<details><summary>The MOSS prompt it compiles to (R0's positive)</summary>

```json
{
  "summary": {
    "caption": "A narrow back alley at night, photo, wet cobblestones, a single street lamp."
  },
  "items": [
    {
      "label": "object",
      "caption": "a heavy green metal door with a rusty handle",
      "bbox": [
        -256,
        -461,
        359,
        -235
      ]
    },
    {
      "label": "text",
      "caption": "a pink neon sign",
      "text": "OPEN 24H",
      "palette": [
        "#FF4FA3"
      ],
      "bbox": [
        -431,
        -185,
        -286,
        185
      ]
    },
    {
      "label": "object",
      "caption": "a red vintage bicycle leaning on the wall",
      "bbox": [
        51,
        -175,
        379,
        134
      ]
    },
    {
      "label": "object",
      "caption": "a dented silver metal bin",
      "bbox": [
        225,
        204,
        451,
        431
      ]
    },
    {
      "label": "object",
      "caption": "a black cat sitting on a metal bin",
      "bbox": [
        81,
        225,
        246,
        390
      ]
    },
    {
      "label": "object",
      "caption": "an old cast-iron street lamp, glowing orange",
      "bbox": [
        -472,
        286,
        -163,
        431
      ]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

</details>

What a box reads under Regions Apply, for example **Lamp**:

> A narrow back alley at night, photo, wet cobblestones, a single street lamp. an old cast-iron street lamp, glowing orange.

## Case E — In front of something (overlap)

One box in front of another: Amber in front of a big window. The upper box owns the overlap. Failure shows as the window's content painted over her, or her merging into the view.

**Look for:** Amber stands in front of the window, not inside the view; the skyline and moon stay in the window; the plant stays in the lower left.

| Seed | R0 | R1 | R2 | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

<details><summary>Layout JSON (paste into the Layout node)</summary>

```json
{
  "nynxzLayout": 1,
  "mode": "moss",
  "layout": {
    "mode": "moss",
    "width": 1024,
    "height": 1024,
    "prompt": "Inside a dim apartment at night, photo, warm lamp light.",
    "meta": [],
    "canvas": "",
    "palette": [],
    "boxes": [
      {
        "id": "b1",
        "name": "Window",
        "x": 0.1,
        "y": 0.05,
        "w": 0.8,
        "h": 0.6,
        "kind": "object",
        "desc": "a large window showing a city skyline at night with a full moon",
        "text": "",
        "palette": []
      },
      {
        "id": "b2",
        "name": "Amber",
        "x": 0.3,
        "y": 0.3,
        "w": 0.4,
        "h": 0.7,
        "kind": "character",
        "desc": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She stands with her back to the window, arms crossed, smiling.",
        "text": "",
        "palette": []
      },
      {
        "id": "b3",
        "name": "Plant",
        "x": 0.0,
        "y": 0.55,
        "w": 0.22,
        "h": 0.45,
        "kind": "object",
        "desc": "a tall potted monstera plant",
        "text": "",
        "palette": []
      }
    ]
  }
}
```

</details>

<details><summary>The MOSS prompt it compiles to (R0's positive)</summary>

```json
{
  "summary": {
    "caption": "Inside a dim apartment at night, photo, warm lamp light."
  },
  "items": [
    {
      "label": "object",
      "caption": "a large window showing a city skyline at night with a full moon",
      "bbox": [
        -461,
        -410,
        154,
        410
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She stands with her back to the window, arms crossed, smiling.",
      "bbox": [
        -205,
        -205,
        512,
        205
      ]
    },
    {
      "label": "object",
      "caption": "a tall potted monstera plant",
      "bbox": [
        51,
        -512,
        512,
        -286
      ]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

</details>

What a box reads under Regions Apply, for example **Amber**:

> Inside a dim apartment at night, photo, warm lamp light. Amber, a young woman with a short copper-red bob, green eyes and light freckles, wearing a yellow raincoat over a navy striped sweater. She stands with her back to the window, arms crossed, smiling.

## Case F — Anime storyboard, hard, written well

An eight-panel portrait page with irregular panels: a tall panel spanning two rows, a three-panel bottom row, and an inset (panel 8) on top of panel 1's corner. Amber and Kai recur in different shots: tiny in a crowd, an extreme close-up of a hand, over the shoulder with Kai in the foreground and Amber far behind. Panel numbers are their own small text boxes, there are three speech bubbles, and a gutter box sits at the bottom of the stack. Each panel box holds the shot and place; each character box holds the identity word for word plus that panel's action.

**Look for:** Count per panel: every panel shows its own beat and no other's; Amber and Kai each appear only in the panels that name them, once, in their boxes; the same face, hair and yukata (Amber) or jinbei (Kai) everywhere; the inset (panel 8) sits on top of panel 1 and shows only the gloved hand; panel 5 shows a hand, not a whole Amber; panel 4 has Kai's back in front and a small Amber far behind; the bubbles read MY KEYCHAIN!, Wait up... and We got it! in the right panels; the numbers 1-8 are legible and in the right panels.

| Seed | R0 | R1 | R2 | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

<details><summary>Layout JSON (paste into the Layout node)</summary>

```json
{
  "nynxzLayout": 1,
  "mode": "moss",
  "layout": {
    "mode": "moss",
    "width": 1024,
    "height": 1536,
    "prompt": "Anime style, cel shading, clean lineart, vivid colours.",
    "meta": [],
    "canvas": "multi_panel",
    "palette": [],
    "boxes": [
      {
        "id": "b0",
        "name": "Gutters",
        "x": 0.0,
        "y": 0.0,
        "w": 1.0,
        "h": 1.0,
        "kind": "object",
        "desc": "white gutters and thick black panel borders",
        "text": "",
        "palette": []
      },
      {
        "id": "b1",
        "name": "Panel 1",
        "x": 0.015,
        "y": 0.015,
        "w": 0.97,
        "h": 0.235,
        "kind": "object",
        "desc": "Panel 1, wide establishing shot at night: a summer festival street packed with paper lanterns, food stalls and a crowd.",
        "text": "",
        "palette": []
      },
      {
        "id": "b2",
        "name": "Amber 1",
        "x": 0.1896,
        "y": 0.1207,
        "w": 0.0679,
        "h": 0.1175,
        "kind": "character",
        "desc": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. Tiny in the crowd, holding a cotton candy, happy.",
        "text": "",
        "palette": []
      },
      {
        "id": "b3",
        "name": "No. 1",
        "x": 0.025,
        "y": 0.023,
        "w": 0.045,
        "h": 0.03,
        "kind": "text",
        "desc": "a small panel number in a white square",
        "text": "1",
        "palette": []
      },
      {
        "id": "b4",
        "name": "Panel 8",
        "x": 0.68,
        "y": 0.14,
        "w": 0.27,
        "h": 0.105,
        "kind": "object",
        "desc": "Panel 8, a round inset panel, extreme close-up: a gloved hand slipping away with a silver cat-shaped keychain.",
        "text": "",
        "palette": []
      },
      {
        "id": "b5",
        "name": "No. 8",
        "x": 0.69,
        "y": 0.148,
        "w": 0.045,
        "h": 0.03,
        "kind": "text",
        "desc": "a small panel number in a white square",
        "text": "8",
        "palette": []
      },
      {
        "id": "b6",
        "name": "Panel 2",
        "x": 0.015,
        "y": 0.265,
        "w": 0.6,
        "h": 0.255,
        "kind": "object",
        "desc": "Panel 2, medium shot: a food stall with a striped awning, lanterns behind.",
        "text": "",
        "palette": []
      },
      {
        "id": "b7",
        "name": "Kai 2",
        "x": 0.345,
        "y": 0.3033,
        "w": 0.21,
        "h": 0.2167,
        "kind": "character",
        "desc": "Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Behind her, holding two red candy apples, puzzled.",
        "text": "",
        "palette": []
      },
      {
        "id": "b8",
        "name": "Amber 2",
        "x": 0.063,
        "y": 0.2905,
        "w": 0.27,
        "h": 0.2295,
        "kind": "character",
        "desc": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. She gasps and points off-panel, her other hand on an empty bag strap, shocked.",
        "text": "",
        "palette": []
      },
      {
        "id": "b9",
        "name": "Bubble 2",
        "x": 0.345,
        "y": 0.2777,
        "w": 0.24,
        "h": 0.0561,
        "kind": "text",
        "desc": "a white speech bubble with a black outline",
        "text": "MY KEYCHAIN!",
        "palette": []
      },
      {
        "id": "b10",
        "name": "No. 2",
        "x": 0.025,
        "y": 0.273,
        "w": 0.045,
        "h": 0.03,
        "kind": "text",
        "desc": "a small panel number in a white square",
        "text": "2",
        "palette": []
      },
      {
        "id": "b11",
        "name": "Panel 3",
        "x": 0.63,
        "y": 0.265,
        "w": 0.355,
        "h": 0.515,
        "kind": "object",
        "desc": "Panel 3, tall shot looking down a stone stairway lined with lanterns, motion lines, light streaks.",
        "text": "",
        "palette": []
      },
      {
        "id": "b12",
        "name": "Amber 3",
        "x": 0.6833,
        "y": 0.4453,
        "w": 0.2485,
        "h": 0.2833,
        "kind": "character",
        "desc": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. She runs down the stairs, yukata sleeves flying, determined.",
        "text": "",
        "palette": []
      },
      {
        "id": "b13",
        "name": "No. 3",
        "x": 0.64,
        "y": 0.273,
        "w": 0.045,
        "h": 0.03,
        "kind": "text",
        "desc": "a small panel number in a white square",
        "text": "3",
        "palette": []
      },
      {
        "id": "b14",
        "name": "Panel 4",
        "x": 0.015,
        "y": 0.535,
        "w": 0.6,
        "h": 0.245,
        "kind": "object",
        "desc": "Panel 4, over-the-shoulder shot past a goldfish-scooping stand with blue water tubs.",
        "text": "",
        "palette": []
      },
      {
        "id": "b15",
        "name": "Amber 4",
        "x": 0.387,
        "y": 0.584,
        "w": 0.108,
        "h": 0.1103,
        "kind": "character",
        "desc": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. Far ahead, small, leaping over the goldfish stand.",
        "text": "",
        "palette": []
      },
      {
        "id": "b16",
        "name": "Kai 4",
        "x": 0.015,
        "y": 0.5963,
        "w": 0.252,
        "h": 0.1837,
        "kind": "character",
        "desc": "Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Seen from behind in the foreground, one arm reaching out.",
        "text": "",
        "palette": []
      },
      {
        "id": "b17",
        "name": "No. 4",
        "x": 0.025,
        "y": 0.543,
        "w": 0.045,
        "h": 0.03,
        "kind": "text",
        "desc": "a small panel number in a white square",
        "text": "4",
        "palette": []
      },
      {
        "id": "b18",
        "name": "Panel 5",
        "x": 0.015,
        "y": 0.795,
        "w": 0.31,
        "h": 0.19,
        "kind": "object",
        "desc": "Panel 5, extreme close-up on a wooden railing.",
        "text": "",
        "palette": []
      },
      {
        "id": "b19",
        "name": "Amber 5",
        "x": 0.0615,
        "y": 0.833,
        "w": 0.2325,
        "h": 0.133,
        "kind": "character",
        "desc": "Amber's hand in a yellow sunflower-patterned yukata sleeve, grabbing a silver cat-shaped keychain.",
        "text": "",
        "palette": []
      },
      {
        "id": "b20",
        "name": "No. 5",
        "x": 0.025,
        "y": 0.803,
        "w": 0.045,
        "h": 0.03,
        "kind": "text",
        "desc": "a small panel number in a white square",
        "text": "5",
        "palette": []
      },
      {
        "id": "b21",
        "name": "Panel 6",
        "x": 0.345,
        "y": 0.795,
        "w": 0.31,
        "h": 0.19,
        "kind": "object",
        "desc": "Panel 6, medium shot of a quiet side path, dropped candy apples on the ground.",
        "text": "",
        "palette": []
      },
      {
        "id": "b22",
        "name": "Kai 6",
        "x": 0.407,
        "y": 0.8235,
        "w": 0.2015,
        "h": 0.1615,
        "kind": "character",
        "desc": "Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Bent over, hands on his knees, panting and sweating.",
        "text": "",
        "palette": []
      },
      {
        "id": "b23",
        "name": "Bubble 6",
        "x": 0.5155,
        "y": 0.8045,
        "w": 0.124,
        "h": 0.0418,
        "kind": "text",
        "desc": "a white speech bubble with a black outline",
        "text": "Wait up...",
        "palette": []
      },
      {
        "id": "b24",
        "name": "No. 6",
        "x": 0.355,
        "y": 0.803,
        "w": 0.045,
        "h": 0.03,
        "kind": "text",
        "desc": "a small panel number in a white square",
        "text": "6",
        "palette": []
      },
      {
        "id": "b25",
        "name": "Panel 7",
        "x": 0.675,
        "y": 0.795,
        "w": 0.31,
        "h": 0.19,
        "kind": "object",
        "desc": "Panel 7, wide shot of a hilltop at night, fireworks bursting in the sky.",
        "text": "",
        "palette": []
      },
      {
        "id": "b26",
        "name": "Amber 7",
        "x": 0.7122,
        "y": 0.852,
        "w": 0.1178,
        "h": 0.133,
        "kind": "character",
        "desc": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. Laughing, holding the keychain up to the fireworks.",
        "text": "",
        "palette": []
      },
      {
        "id": "b27",
        "name": "Kai 7",
        "x": 0.8362,
        "y": 0.8425,
        "w": 0.1178,
        "h": 0.1425,
        "kind": "character",
        "desc": "Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Laughing beside her, rubbing the back of his head.",
        "text": "",
        "palette": []
      },
      {
        "id": "b28",
        "name": "Bubble 7",
        "x": 0.8455,
        "y": 0.8045,
        "w": 0.124,
        "h": 0.0418,
        "kind": "text",
        "desc": "a white speech bubble with a black outline",
        "text": "We got it!",
        "palette": []
      },
      {
        "id": "b29",
        "name": "No. 7",
        "x": 0.685,
        "y": 0.803,
        "w": 0.045,
        "h": 0.03,
        "kind": "text",
        "desc": "a small panel number in a white square",
        "text": "7",
        "palette": []
      }
    ]
  }
}
```

</details>

<details><summary>The MOSS prompt it compiles to (R0's positive)</summary>

```json
{
  "summary": {
    "canvas": "multi_panel",
    "caption": "Anime style, cel shading, clean lineart, vivid colours."
  },
  "items": [
    {
      "label": "object",
      "caption": "white gutters and thick black panel borders",
      "bbox": [
        -768,
        -512,
        768,
        512
      ]
    },
    {
      "label": "object",
      "caption": "Panel 1, wide establishing shot at night: a summer festival street packed with paper lanterns, food stalls and a crowd.",
      "bbox": [
        -745,
        -497,
        -384,
        497
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. Tiny in the crowd, holding a cotton candy, happy.",
      "bbox": [
        -583,
        -318,
        -402,
        -248
      ]
    },
    {
      "label": "text",
      "caption": "a small panel number in a white square",
      "text": "1",
      "bbox": [
        -733,
        -487,
        -686,
        -440
      ]
    },
    {
      "label": "object",
      "caption": "Panel 8, a round inset panel, extreme close-up: a gloved hand slipping away with a silver cat-shaped keychain.",
      "bbox": [
        -553,
        184,
        -391,
        461
      ]
    },
    {
      "label": "text",
      "caption": "a small panel number in a white square",
      "text": "8",
      "bbox": [
        -541,
        194,
        -494,
        241
      ]
    },
    {
      "label": "object",
      "caption": "Panel 2, medium shot: a food stall with a striped awning, lanterns behind.",
      "bbox": [
        -361,
        -497,
        31,
        118
      ]
    },
    {
      "label": "character",
      "caption": "Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Behind her, holding two red candy apples, puzzled.",
      "bbox": [
        -303,
        -159,
        31,
        57
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. She gasps and points off-panel, her other hand on an empty bag strap, shocked.",
      "bbox": [
        -322,
        -448,
        31,
        -171
      ]
    },
    {
      "label": "text",
      "caption": "a white speech bubble with a black outline",
      "text": "MY KEYCHAIN!",
      "bbox": [
        -342,
        -159,
        -255,
        88
      ]
    },
    {
      "label": "text",
      "caption": "a small panel number in a white square",
      "text": "2",
      "bbox": [
        -349,
        -487,
        -302,
        -440
      ]
    },
    {
      "label": "object",
      "caption": "Panel 3, tall shot looking down a stone stairway lined with lanterns, motion lines, light streaks.",
      "bbox": [
        -361,
        133,
        431,
        497
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. She runs down the stairs, yukata sleeves flying, determined.",
      "bbox": [
        -85,
        187,
        352,
        443
      ]
    },
    {
      "label": "text",
      "caption": "a small panel number in a white square",
      "text": "3",
      "bbox": [
        -349,
        143,
        -302,
        190
      ]
    },
    {
      "label": "object",
      "caption": "Panel 4, over-the-shoulder shot past a goldfish-scooping stand with blue water tubs.",
      "bbox": [
        53,
        -497,
        431,
        118
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. Far ahead, small, leaping over the goldfish stand.",
      "bbox": [
        129,
        -116,
        299,
        -5
      ]
    },
    {
      "label": "character",
      "caption": "Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Seen from behind in the foreground, one arm reaching out.",
      "bbox": [
        147,
        -497,
        431,
        -238
      ]
    },
    {
      "label": "text",
      "caption": "a small panel number in a white square",
      "text": "4",
      "bbox": [
        66,
        -487,
        113,
        -440
      ]
    },
    {
      "label": "object",
      "caption": "Panel 5, extreme close-up on a wooden railing.",
      "bbox": [
        453,
        -497,
        745,
        -179
      ]
    },
    {
      "label": "character",
      "caption": "Amber's hand in a yellow sunflower-patterned yukata sleeve, grabbing a silver cat-shaped keychain.",
      "bbox": [
        511,
        -450,
        716,
        -210
      ]
    },
    {
      "label": "text",
      "caption": "a small panel number in a white square",
      "text": "5",
      "bbox": [
        465,
        -487,
        512,
        -440
      ]
    },
    {
      "label": "object",
      "caption": "Panel 6, medium shot of a quiet side path, dropped candy apples on the ground.",
      "bbox": [
        453,
        -159,
        745,
        159
      ]
    },
    {
      "label": "character",
      "caption": "Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Bent over, hands on his knees, panting and sweating.",
      "bbox": [
        496,
        -96,
        745,
        112
      ]
    },
    {
      "label": "text",
      "caption": "a white speech bubble with a black outline",
      "text": "Wait up...",
      "bbox": [
        467,
        15,
        532,
        143
      ]
    },
    {
      "label": "text",
      "caption": "a small panel number in a white square",
      "text": "6",
      "bbox": [
        465,
        -149,
        512,
        -102
      ]
    },
    {
      "label": "object",
      "caption": "Panel 7, wide shot of a hilltop at night, fireworks bursting in the sky.",
      "bbox": [
        453,
        179,
        745,
        497
      ]
    },
    {
      "label": "character",
      "caption": "Amber, a teenage girl with a short copper-red bob, big green eyes and light freckles, wearing a yellow yukata with a sunflower pattern and a red obi. Laughing, holding the keychain up to the fireworks.",
      "bbox": [
        540,
        217,
        745,
        338
      ]
    },
    {
      "label": "character",
      "caption": "Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Laughing beside her, rubbing the back of his head.",
      "bbox": [
        526,
        344,
        745,
        465
      ]
    },
    {
      "label": "text",
      "caption": "a white speech bubble with a black outline",
      "text": "We got it!",
      "bbox": [
        467,
        353,
        532,
        481
      ]
    },
    {
      "label": "text",
      "caption": "a small panel number in a white square",
      "text": "7",
      "bbox": [
        465,
        189,
        512,
        236
      ]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

</details>

What a box reads under Regions Apply, for example **Kai 7**:

> Anime style, cel shading, clean lineart, vivid colours. Kai, a tall teenage boy with straight black chin-length hair, brown eyes and a thin scar over his left eyebrow, wearing a dark blue jinbei. Laughing beside her, rubbing the back of his head.

## Case G — The same storyboard, written the reported way

Panel F's geometry and story, written the way the reported layout was: one box per panel, every box opening with the same sentence, the number described in prose, Kai only named in passing. Compare it with F under the same seeds and runs.

**Look for:** The same checklist as F. Expect worse: G is the baseline that shows how much of F's result comes from the writing and how much from Regions Apply.

| Seed | R0 | R1 | R2 | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

<details><summary>Layout JSON (paste into the Layout node)</summary>

```json
{
  "nynxzLayout": 1,
  "mode": "moss",
  "layout": {
    "mode": "moss",
    "width": 1024,
    "height": 1536,
    "prompt": "Anime style, cel shading.",
    "meta": [],
    "canvas": "multi_panel",
    "palette": [],
    "boxes": [
      {
        "id": "b0",
        "name": "Background",
        "x": 0.0,
        "y": 0.0,
        "w": 1.0,
        "h": 1.0,
        "kind": "object",
        "desc": "White background, between frames",
        "text": "",
        "palette": []
      },
      {
        "id": "b1",
        "name": "Frame 1",
        "x": 0.015,
        "y": 0.015,
        "w": 0.97,
        "h": 0.235,
        "kind": "object",
        "desc": "A frame featuring Amber. The number \"1\" is drawn in the corner. She holds a cotton candy at a night festival.",
        "text": "",
        "palette": []
      },
      {
        "id": "b2",
        "name": "Frame 8",
        "x": 0.68,
        "y": 0.14,
        "w": 0.27,
        "h": 0.105,
        "kind": "object",
        "desc": "A frame featuring Amber. The number \"8\" is drawn in the corner. A hand steals her keychain.",
        "text": "",
        "palette": []
      },
      {
        "id": "b3",
        "name": "Frame 2",
        "x": 0.015,
        "y": 0.265,
        "w": 0.6,
        "h": 0.255,
        "kind": "object",
        "desc": "A frame featuring Amber. The number \"2\" is drawn in the corner. She gasps and points.",
        "text": "",
        "palette": []
      },
      {
        "id": "b4",
        "name": "Frame 3",
        "x": 0.63,
        "y": 0.265,
        "w": 0.355,
        "h": 0.515,
        "kind": "object",
        "desc": "A frame featuring Amber. The number \"3\" is drawn in the corner. She runs down a stairway.",
        "text": "",
        "palette": []
      },
      {
        "id": "b5",
        "name": "Frame 4",
        "x": 0.015,
        "y": 0.535,
        "w": 0.6,
        "h": 0.245,
        "kind": "object",
        "desc": "A frame featuring Amber. The number \"4\" is drawn in the corner. She leaps over a goldfish stand.",
        "text": "",
        "palette": []
      },
      {
        "id": "b6",
        "name": "Frame 5",
        "x": 0.015,
        "y": 0.795,
        "w": 0.31,
        "h": 0.19,
        "kind": "object",
        "desc": "A frame featuring Amber. The number \"5\" is drawn in the corner. She grabs the keychain.",
        "text": "",
        "palette": []
      },
      {
        "id": "b7",
        "name": "Frame 6",
        "x": 0.345,
        "y": 0.795,
        "w": 0.31,
        "h": 0.19,
        "kind": "object",
        "desc": "A frame featuring Amber. The number \"6\" is drawn in the corner. Kai is panting.",
        "text": "",
        "palette": []
      },
      {
        "id": "b8",
        "name": "Frame 7",
        "x": 0.675,
        "y": 0.795,
        "w": 0.31,
        "h": 0.19,
        "kind": "object",
        "desc": "A frame featuring Amber. The number \"7\" is drawn in the corner. They laugh under fireworks.",
        "text": "",
        "palette": []
      }
    ]
  }
}
```

</details>

<details><summary>The MOSS prompt it compiles to (R0's positive)</summary>

```json
{
  "summary": {
    "canvas": "multi_panel",
    "caption": "Anime style, cel shading."
  },
  "items": [
    {
      "label": "object",
      "caption": "White background, between frames",
      "bbox": [
        -768,
        -512,
        768,
        512
      ]
    },
    {
      "label": "object",
      "caption": "A frame featuring Amber. The number \"1\" is drawn in the corner. She holds a cotton candy at a night festival.",
      "bbox": [
        -745,
        -497,
        -384,
        497
      ]
    },
    {
      "label": "object",
      "caption": "A frame featuring Amber. The number \"8\" is drawn in the corner. A hand steals her keychain.",
      "bbox": [
        -553,
        184,
        -391,
        461
      ]
    },
    {
      "label": "object",
      "caption": "A frame featuring Amber. The number \"2\" is drawn in the corner. She gasps and points.",
      "bbox": [
        -361,
        -497,
        31,
        118
      ]
    },
    {
      "label": "object",
      "caption": "A frame featuring Amber. The number \"3\" is drawn in the corner. She runs down a stairway.",
      "bbox": [
        -361,
        133,
        431,
        497
      ]
    },
    {
      "label": "object",
      "caption": "A frame featuring Amber. The number \"4\" is drawn in the corner. She leaps over a goldfish stand.",
      "bbox": [
        53,
        -497,
        431,
        118
      ]
    },
    {
      "label": "object",
      "caption": "A frame featuring Amber. The number \"5\" is drawn in the corner. She grabs the keychain.",
      "bbox": [
        453,
        -497,
        745,
        -179
      ]
    },
    {
      "label": "object",
      "caption": "A frame featuring Amber. The number \"6\" is drawn in the corner. Kai is panting.",
      "bbox": [
        453,
        -159,
        745,
        159
      ]
    },
    {
      "label": "object",
      "caption": "A frame featuring Amber. The number \"7\" is drawn in the corner. They laugh under fireworks.",
      "bbox": [
        453,
        179,
        745,
        497
      ]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

</details>

What a box reads under Regions Apply, for example **Frame 7**:

> Anime style, cel shading. A frame featuring Amber. The number "7" is drawn in the corner. They laugh under fireworks.
