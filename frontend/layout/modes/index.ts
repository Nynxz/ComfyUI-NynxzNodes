// Every layout mode, registered so a layout copied from one can be pasted into another.

import { registerMode } from "../mode";
import { PROMPT_MODES } from "./prompt";
import { regionsMode } from "./regions";

for (const mode of [regionsMode, ...PROMPT_MODES]) registerMode(mode);

export { regionsMode, PROMPT_MODES };
