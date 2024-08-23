# <img src='https://raw.githack.com/FortAwesome/Font-Awesome/master/svgs/solid/robot.svg' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> HuggingChat Persona

Talk to HuggingChat through OpenVoiceOS.

Uses [Hugging Chat](https://huggingface.co/chat/) via [hugchat](https://github.com/Soulter/hugging-chat-api) to create some fun interactions.  Phrases not explicitly handled by other skills will be run by a LLM, so nearly every interaction will have _some_ response.

## Usage

```python
from ovos_solver_hugchat_persona import HuggingChatPersonaSolver

bot = HuggingChatPersonaSolver(
    {
        "email": "{your-hf-email}",
        "password": "{your-hf-password}",
        "persona": "helpful, creative, clever, and very friendly"
    }
)
print(bot.get_spoken_answer("describe quantum mechanics in simple terms"))
# Quantum mechanics is a branch of physics that deals with the behavior of particles on a very small scale, such as atoms and subatomic particles. It explores the idea that particles can exist in multiple states at once and that their behavior is not predictable in the traditional sense.
print(bot.get_spoken_answer("Quem encontrou o caminho maritimo para o Brasil"))
# Explorador português Pedro Álvares Cabral é creditado com a descoberta do Brasil em 1500

```

This plugin will work with [ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server)

## Configuration

This plugin can be configured as follows

```json
{
    "email": 'your_hf_email',
    "password": 'your_hf_password',
    "initial_prompt": "You are a helpful assistant."
}
```

