from openai import OpenAI

class API:
    """
    Represents an API client for interacting with a remote service.
    Using a class to represent the API client allows for easy extensibility.

    Attributes:
        api_key (str): The API key for authentication.
        base_url (str): The base URL for the API.

    Methods:
        __init__(api_key, base_url): Initializes an instance of the API class.
        get_openai(): Returns an instance of the OpenAI class.

    """

    def __init__(self, api_key, base_url):
            """
            Initializes an instance of the API class.

            Args:
                api_key (str): The API key used for authentication.
                base_url (str): The base URL of the API.

            Returns:
                None
            """
            self.api_key = api_key
            self.base_url = base_url

    def get_openai(self):
        """
        Returns:
            An instance of the OpenAI class.
        """
        return OpenAI(api_key=self.api_key,
                      base_url=self.base_url)