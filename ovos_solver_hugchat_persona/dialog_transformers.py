from typing import Optional, Dict, Tuple, Any

from ovos_plugin_manager.templates.transformers import DialogTransformer

from ovos_solver_hugchat_persona import HuggingChatCompletionsSolver


class HuggingChatDialogTransformer(DialogTransformer):
    def __init__(
            self, name : str = "ovos-dialog-transformer-hugchat-plugin",
            priority : int = 10,
            config : Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, priority, config)
        self.solver = HuggingChatCompletionsSolver({
            "email": self.config.get("key"),
            "password": self.config.get("password"),
            "enable_memory": False,
            "initial_prompt": "your task is to rewrite text as if it was spoken by a different character"
        })

    def transform(self, dialog: str, context: dict = None) -> Tuple[str, dict]:
        """
        Optionally transform passed dialog and/or return additional context
        :param dialog: str utterance to mutate before TTS
        :returns: str mutated dialog
        """
        prompt = context.get("prompt") or self.config.get("rewrite_prompt")
        if not prompt:
            return dialog, context
        return self.solver.get_spoken_answer(f"{prompt} : {dialog}"), context
