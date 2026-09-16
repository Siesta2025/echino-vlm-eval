

class Provider:
    def __init__(self, name: str, api_key: str = None, model_list: list = None):
        self.name = name
        self.api_key = api_key
        self.model_list = model_list or []

class OpenAIProvider(Provider):
    def __init__(self, name: str, api_key: str, model_list: list = None):
        super().__init__(name, api_key, model_list)

class QwenProvider(Provider):
    def __init__(self, name: str, api_key: str, model_list: list = None):
        super().__init__(name, api_key, model_list)

class GoogleProvider(Provider):
    def __init__(self, name: str, api_key: str, model_list: list = None):
        super().__init__(name, api_key, model_list)

class Model:
    def __init__(self, model_name: str, provider: Provider):
        self.provider = provider