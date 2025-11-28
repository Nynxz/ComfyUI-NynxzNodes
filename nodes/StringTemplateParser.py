from comfy_api.latest import io, ui
import re
import json


class StringTemplateParser(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="nynxz.Parse.StringTemplate",
            display_name="String Template Parser",
            category="Nynxz",
            inputs=[
                io.String.Input("source_text", default=""),
                io.String.Input("variable_name", default=""),
                io.String.Input("replacement_text", default=""),
                io.String.Input("replacements_json",
                                default="", multiline=True),
                io.String.Input("default_for_missing", default=""),
                io.Boolean.Input(
                    "replace_missing_with_default", default=False),

            ],
            outputs=[
                io.String.Output("result"),
            ]
        )

    @classmethod
    def execute(cls, source_text, variable_name, replacement_text, replacements_json, default_for_missing, replace_missing_with_default):
        """Replace placeholders in the form {{name}} inside source_text.

        Usage modes:
        - Single replacement: provide variable_name and replacement_text
        - Batch replacements: provide replacements_json (JSON object string)

        Behavior:
        - If a replacements_json object is provided it takes precedence and is
          applied to all placeholders.
        - If a single variable_name is provided it will replace matching
          placeholder(s) with replacement_text.
        - If a placeholder is missing in replacements and replace_missing_with_default
          is True, the placeholder will be replaced with default_for_missing. If
          False, missing placeholders are left unchanged.
        """

        # fast path: nothing to do
        if not source_text:
            return io.NodeOutput(source_text)

        # build mapping from provided JSON if available
        mapping = {}
        if replacements_json:
            try:
                parsed = json.loads(replacements_json)
                if isinstance(parsed, dict):
                    mapping.update({str(k): str(v) for k, v in parsed.items()})
            except Exception:
                # invalid JSON — ignore mapping
                mapping = {}

        # single replacement if provided
        if variable_name and replacement_text and variable_name not in mapping:
            mapping[str(variable_name)] = str(replacement_text)

        # regex to find placeholders like {{ name }}
        pattern = re.compile(r"\{\{\s*([A-Za-z0-9_\-]+)\s*\}\}")

        def _repl(m):
            name = m.group(1)
            if name in mapping:
                return mapping[name]
            if replace_missing_with_default:
                return default_for_missing or ""
            return m.group(0)  # leave untouched

        result = pattern.sub(_repl, source_text)

        return io.NodeOutput(result)
