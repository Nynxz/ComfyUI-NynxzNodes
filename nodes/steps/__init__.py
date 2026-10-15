"""Step films — watch a sampler work, one frame per step.

    Load Diffusion Model ──> Record Steps ──> MODEL ──> KSampler / SamplerCustom ──> LATENT
                                   └──> steps ──────────────> Step Film <──────────────┘
                                                                  └──> IMAGE ──> Create Video

`Record Steps` patches the model so every sampler that runs on it hands over what it predicted at
each step; `Step Film` decodes those and lays them out as frames with a progress bar underneath. A
second recording wired into `Step Film` plays beside the first on the same clock, which is how two
samplers (SoftPaint against Differential Diffusion, say) are compared step for step.

The hook is a sampler wrapper on the model, not a sampler of its own, so it records any sampler
and any scheduler, KSampler included, and costs nothing while sampling. It takes one frame per
outer step from the sampler's preview callback: SoftPaint's inner Langevin calls don't add frames.
"""
