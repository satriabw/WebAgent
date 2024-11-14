class Agent:
    """Base Agent class"""
    async def call_agent(self, *args, **kwargs):
        raise NotImplementedError
