import os
import json
from typing import Optional, Dict, Generator, Any

import requests
from hugchat import hugchat
from hugchat.login import Login

from ovos_plugin_manager.templates.solvers import QuestionSolver
from ovos_utils.log import LOG


COOKIE_PATH_TEMPLATE = "~/.ovos/cookies/{email}.json"


class HuggingChatCompletionsSolver(QuestionSolver):
    enable_tx = False
    priority = 25

    def __init__(self, config : Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.email = self.config.get("email")
        if not self.email:
            LOG.error("'email' not set in config")
            raise ValueError("email must be set")
        self.password = self.config.get("password")
        if not self.password:
            LOG.error("'password' not set in config")
            raise ValueError("password must be set")
        self.web_search = self.config.get("web_search", False)
        self.cookies = self._authenticate()
        # Create your ChatBot
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict())
        self.available_models = list(
            map(lambda m: m.__str__().lower(), self.chatbot.get_available_llm_models())
        )
        self.engine = self.config.get("model", self.available_models[0])
        self._select_model()
        self.initial_prompt = config.get("initial_prompt", "You are a helpful assistant.")

    def _select_model(self):
        """Select Hugging Chat model to be used."""
        model_id = 0
        models_ids = list(range(len(self.available_models)))
        for i, model_name in enumerate(self.available_models):
            if self.engine in model_name:
                model_id = i
                del models_ids[i]
                break
        LOG.debug(f"available models: {self.available_models}")
        while models_ids:  # while there are models left to try
            try:
                self.chatbot.switch_llm(model_id)
                LOG.debug(f"selected model {model_id}-{self.available_models[model_id]}")
                break
            except Exception as e:
                LOG.debug(f"unable to selected {model_id}-{self.available_models[model_id]}: {e}")
                if not models_ids:
                    LOG.error(f"unable to selected {model_id}-{self.available_models[model_id]}: {e}")
                    raise RuntimeError("unable to select any model!")
            # Try next
            model_id = models_ids.pop(0)


    def _authenticate(self) -> requests.sessions.RequestsCookieJar:
        """Authenticate user for a Hugging Chat session and retrieve cookie jar."""
        cookie_path = os.path.expanduser(COOKIE_PATH_TEMPLATE.format(email=self.email))
        cookie_base_dir = os.path.dirname(cookie_path)
        auth = Login(self.email, self.password)
        if not os.path.exists(cookie_path):
            os.makedirs(cookie_base_dir, exist_ok=True)
            cookies = auth.login(cookie_dir_path=cookie_base_dir, save_cookies=True)
        else:
            try:
                with open(cookie_path, "r", encoding="utf-8") as f:
                    cookies_dict : dict = json.load(f)
                    _ = list(cookies_dict.keys()).index('token')
                    _ = list(cookies_dict.keys()).index('hf-chat')
                    cookies = auth.load_cookies(cookie_base_dir)                        
            except Exception as e:
                LOG.error('error retrieving cookies... retrying authentication')
                cookies = auth.login(cookie_dir_path=cookie_base_dir, save_cookies=True)
        return cookies

    # Hugging Chat integration
    def _do_api_request(self, prompt: str) -> str:
        """Send query to ChatBot"""
        response = self.chatbot.chat(prompt, web_search=self.web_search)
        return response["text"]

    def _do_streaming_api_request(self, prompt: str) -> Generator[str, None, None]:
        """Send query to ChatBot"""
        for chunk in self.chatbot.query(
            prompt,
            web_search=self.web_search,
            stream=True
        ):
            if chunk:
                yield chunk["token"]

    def get_spoken_answer(self, query: str, **kwargs):
        response = self._do_api_request(query)
        answer = response.strip()
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        return answer

    # officially exported Solver methods
    def stream_utterances(self, query) -> Generator[str, None, None]:
        answer = ""
        for chunk in self._do_streaming_api_request(query):
            answer += chunk
            if any(chunk.endswith(p) for p in [".", "!", "?", "\n", ":"]):
                if answer.strip():
                    yield answer
                answer = ""


# Base models
class CommandRPlus(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "command-r-plus"
        super().__init__(config=config)


class Llama3(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "llama-3"
        super().__init__(config=config)


class ZephyrOrpo(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "zephyr-orpo"
        super().__init__(config=config)


class Mixtral(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "mixtral"
        super().__init__(config=config)


class NousHermes2(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "nous-hermes-2"
        super().__init__(config=config)


class Yi(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "yi"
        super().__init__(config=config)


class Gemma(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "gemma"
        super().__init__(config=config)


class Mistral(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "mistral"
        super().__init__(config=config)


# Code completion
class Phi3(HuggingChatCompletionsSolver):
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        config = config or {}
        config["model"] = "phi-3"
        super().__init__(config=config)
