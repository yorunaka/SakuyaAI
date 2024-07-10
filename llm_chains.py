from prompt_templates import memory_prompt_template
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
# from utils import load_config
import chromadb
import yaml

# config = load_config()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_normal_chain(chat_history):
    return chatChain(chat_history)

def create_llm(model_path = config["ctransformers"]["model_path"], model_type = config["ctransformers"]["model_type"], model_config = config["ctransformers"]["model_config"]):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm

def create_embeddings(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(embeddings_path)

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key = "History", chat_memory = chat_history, k = 3)

def create_llm_chains(llm, chat_prompt, memory):
    return LLMChain(llm = llm, prompt = chat_prompt, memory = memory)


def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)


class chatChain:
    def __init__(self, chat_history):
        llm = create_llm()
        self.memory = create_chat_memory(chat_history)
        self.llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chains(llm, chat_prompt)

    def run(self, user_input):
        return self.llm_chain.run(human_input = user_input,stop = ["Human:"])