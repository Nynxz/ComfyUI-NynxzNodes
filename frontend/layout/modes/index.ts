// Every layout mode, registered so a layout copied from one can be pasted into another.
// `PROMPT_MODES` includes the unified `regions` mode; `regionsMode` stays exported for legacy
// Fresco nodes still open in old workflows.

import { registerMode } from "../mode";
import { PROMPT_MODES } from "./prompt";
import { regionsMode } from "./regions";

for (const mode of PROMPT_MODES) registerMode(mode);

export { regionsMode, PROMPT_MODES };
