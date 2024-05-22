from typing import Optional, Dict, Any
from ovos_solver_hugchat_persona.engines import HuggingChatCompletionsSolver


class HuggingChatPersonaSolver(HuggingChatCompletionsSolver):
    """default "Persona" engine"""

    def __init__(self, config : Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        self.default_persona = config.get("persona") or "helpful, creative, clever, and very friendly"

    def get_prompt(self, utt: str, persona : Optional[str] = None):
        persona = persona or self.config.get("persona") or self.default_persona
        initial_prompt = (
            "You are a helpful assistant. "
            "You understand all languages. "
            "You give short and factual answers. "
            "You answer in the same language the question was asked. "
            f"You are {persona}."
        )
        prompt = f"{initial_prompt}\n\n{utt}\n"
        return prompt

    # officially exported Solver methods
    def get_spoken_answer(self, query: str, **kwargs):
        context = context or {}
        persona = context.get("persona") or self.default_persona
        prompt = self.get_prompt(query, persona)
        response = self._do_api_request(prompt)
        answer = response.strip()
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        return answer


if __name__ == "__main__":
    bot = HuggingChatPersonaSolver({"email": "your-hf-email", "password": "your-hf-password"})
    for utt in bot.stream_utterances("describe quantum mechanics in simple terms"):
        print(utt)
        #  Quantum mechanics is a branch of physics that studies the behavior of atoms and particles at the smallest scales.
        #  It describes how these particles interact with each other, move, and change energy levels.
        #  Think of it like playing with toy building blocks that represent particles.
        #  Instead of rigid structures, these particles can be in different energy levels or "states." Quantum mechanics helps scientists understand and predict these states, making it crucial for many fields like chemistry, materials science, and engineering.


    # Quantum mechanics is a branch of physics that deals with the behavior of particles on a very small scale, such as atoms and subatomic particles. It explores the idea that particles can exist in multiple states at once and that their behavior is not predictable in the traditional sense.
    print(bot.spoken_answer("Quem encontrou o caminho maritimo para o Brasil?"))
    # Explorador português Pedro Álvares Cabral é creditado com a descoberta do Brasil em 1500
