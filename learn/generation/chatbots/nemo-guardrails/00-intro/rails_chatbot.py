import os
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentOutputParser, load_tools, initialize_agent
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish, HumanMessage

from nemoguardrails import LLMRails, RailsConfig
import nest_asyncio
nest_asyncio.apply()


class OutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS
    
    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            # this will work IF the text is a valid JSON with action and action_input
            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": text}, text)
    
    @property
    def _type(self) -> str:
        return "conversational_chat"


def message_to_dict(message: str):
    return {
        'role': message.type,
        'content': message.content
    }

def chat_history_for_nemo(messages: list):
    """Transforms a list of message objects into a list of message dicts
    to suit NeMo Guardrails requirements
    """
    return [message_to_dict(message) for message in messages]

class Guardrails:
    def __init__(
        self,
        agent,
        model_name: Optional[str] = "text-davinci-003"
    ):
        self.yaml_config = f"""
        models:
          - type: main
            engine: openai
            model: {model_name}"""
        self.colang_config = """
        define user mentions politics
            "tell me about politics"
            "what do you think about politics"
            "left wing"
            "right wing"
        define bot politics response
            "I am a shopping assistant, I don't know much about politics."
        define flow politics
            user mentions politics
            bot apologizes for not being able to discuss politics
        """
        # create rails config from above configs
        # (make sure to put in same order, colang first, then yaml)
        config = RailsConfig.from_content(
            self.colang_config, self.yaml_config
        )
        self.rails = LLMRails(config, verbose=True)
        # TODO not sure if below is needed
        self.rails.register_action(action=agent, name="agent")

    async def apply_guardrails(self, chat_history: list):
        # convert chat history to format that NeMo Guardrails expects
        chat_history = chat_history_for_nemo(chat_history)
        print(chat_history)
        assert len(chat_history) >= 1
        print("len(chat_history) > 1")
        # apply guardrails considering the chat history incl. most recent interactions
        response = await self.rails.generate_async(messages=chat_history)
        print(response)
        response = response['content']
        return response


class Chatbot:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0613",
        openai_api_key: str = None,
        verbose: Optional[bool] = False,
    ):
        self.model_name = model_name
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        assert openai_api_key is not None, "Please provide an OpenAI API key"
        # initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.3
        )
        # initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, output_key="output",
            return_messages=True
        )
        # initialize tools for agent
        self.tools = load_tools(["llm-math"], llm=self.llm)
        # initialize output parser for agent
        parser = OutputParser()
        # initialize agent
        self.agent = initialize_agent(
            agent="chat-conversational-react-description",
            tools=self.tools,
            llm=self.llm,
            verbose=verbose,
            early_stopping_method="generate",
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={"output_parser": parser}
        )
        self.guardrails = Guardrails(agent=self.agent)
        # update the prompt with custom system message
        sys_msg = """Assistant is a very helpful shopping assistant named Joey that helps users with any shopping task.

        Joey is very friendly and amicable, driving the conversation forward with the user's shopping plans while still making small talk.
        
        Joey is qualified to help with a wide range of tasks, but is not able to do any math related tasks."""
        new_prompt = self.agent.agent.create_prompt(
            system_message=sys_msg, tools=self.tools
        )
        self.agent.agent.llm_chain.prompt = new_prompt

    async def _call(self, inputs):
        intermediate_steps = None
        res, intermediate_steps = self.agent({'input': inputs})
        response_object = None
        if self.guardrails is not None:
            # get chat history + latest user message
            new_msg = HumanMessage(content=inputs)
            chat_history = self.memory.chat_memory.messages + [new_msg]
            print(chat_history)
            # apply guardrails considering the chat history incl. most recent interactions
            coro = self.guardrails.apply_guardrails(chat_history)
            res, intermediate_steps = await coro
        else:
            # guardrails not enabled, so just process directly via agent
            res, intermediate_steps = self.agent({'input': inputs})
        return res, intermediate_steps
