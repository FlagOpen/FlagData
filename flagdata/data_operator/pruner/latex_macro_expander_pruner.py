import regex as re
from ..base_operator import BaseOperator

class LatexMacroExpanderPruner(BaseOperator):
    """Pruner to expand macro definitions in the document body of LaTeX samples."""

    def __init__(self, text_key='content', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key

    def _build_non_arg_macros_dict(self, file_content):
        # regex for extracting \newcommand macros without arguments
        non_arg_nc_reg = re.compile(
            pattern=r'\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$',
            flags=re.MULTILINE)

        # regex for extracting \def macros without arguments
        non_arg_def_reg = re.compile(
            pattern=r'\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$',
            flags=re.MULTILINE)

        # Extract all user-defined LaTeX macros from the preamble
        macros = {}
        for reg in [non_arg_nc_reg, non_arg_def_reg]:
            for match in reg.finditer(file_content):
                macro_name = match.group(1).encode('unicode-escape').decode('utf-8')
                macro_val = match.group(2).encode('unicode-escape').decode('utf-8')
                macros[macro_name] = macro_val
        return macros

    def process(self, sample):
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        file_content = sample[self.text_key]
        non_arg_macros = self._build_non_arg_macros_dict(file_content)

        # Inline-expand all non-arg macros
        for macro_name, macro_value in non_arg_macros.items():
            sample[self.text_key] = re.sub(
                pattern=r'(' + macro_name + r')' + r'([^a-zA-Z0-9])',
                repl=macro_value + r'\2',
                string=sample[self.text_key])

        # Inline-expand macros that use args (not implemented yet)
        # TODO: Handle macros with arguments

        return sample
