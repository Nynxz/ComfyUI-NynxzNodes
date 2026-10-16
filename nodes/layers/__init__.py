"""Layer tinkering: switch a diffusion transformer's blocks off, for guidance or to see what they do.

    Load Diffusion Model ──> Skip Layers ──> MODEL ──> any sampler

`Skip Layers` works on any model with a list of residual blocks (Flux 3, Krea 2, Flux, Wan…),
including the ones core's Skip Layer Guidance can't reach because their code ignores
`patches_replace`.
"""
