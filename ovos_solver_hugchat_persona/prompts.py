from typing import Optional, Dict, Any
from ovos_solver_hugchat_persona.engines import HuggingChatCompletionsSolver


# Voice Assistant Prompt Engineering
class HuggingChatPersonaPromptSolver(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "llama-3"
        super().__init__(config=config)
        self.memory = config.get("enable_memory", True)
        self.max_utts = config.get("memory_size", 15)
        self.qa_pairs = []  # tuple of q+a
        self.current_q = None
        self.current_a = None
        self.default_persona = config.get("persona") or "helpful, creative, clever, and very friendly"

    def get_chat_history(self, persona : Optional[str] = None):
        if len(self.qa_pairs) > self.max_utts:
            qa = self.qa_pairs[-1 * self.max_utts:]
        else:
            qa = self.qa_pairs

        persona = persona or self.config.get("persona") or self.default_persona
        initial_prompt = (
            "The following is a conversation with an AI assistant. "
            "The assistant understands all languages. "
            "The assistant gives short and factual answers. "
            "The assistant answers in the same language the question was asked. "
            f"The assistant is {persona}."
        )
        chat = f"{initial_prompt}\n\n"
        if qa:
            qa = "\n".join([f"Human: {q}\nAI: {a}" for q, a in qa])
            if chat.endswith("\nHuman: "):
                chat = chat[-1 * len("\nHuman: "):]
            if chat.endswith("\nAI: "):
                chat += f"Please rephrase the question\n"
            chat += qa
        return chat

    def get_prompt(self, utt : str, persona : Optional[str] = None):
        self.current_q = None
        self.current_a = None
        prompt = self.get_chat_history(persona)
        if not prompt.endswith("\nHuman: "):
            prompt += f"\nHuman: {utt}?\nAI: "
        else:
            prompt += f"{utt}?\nAI: "
        return prompt

    # officially exported Solver methods
    def get_spoken_answer(self, query : str, **kwargs):
        context = context or {}
        persona = context.get("persona") or self.default_persona
        prompt = self.get_prompt(query, persona)
        response = self._do_api_request(prompt)
        answer = response.split("Human: ")[0].split("AI: ")[0].strip()
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        if self.memory:
            self.qa_pairs.append((query, answer))
        return answer
