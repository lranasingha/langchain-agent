from langchain import OpenAI, HuggingFaceHub, LLMChain, PromptTemplate


# This class is used to create a model for the agent
# NOTE: set the respective model provider (i.e. OpenAI) API key environment variable before running
class CustomLLM:

    def huggingface_model(self, model_repo_id):
        llm = HuggingFaceHub(repo_id=model_repo_id)
        prompt = """inputs:{code}"""
        prompt_temp = PromptTemplate(template=prompt, input_variables=["code"])
        return LLMChain(llm=llm, prompt=prompt_temp)


if __name__ == '__main__':
    # llm = OpenAI(model_name="text-davinci-003")
    # print(llm("Hello, my name is"))
    custom_llm = CustomLLM()
    code = "def standard_deviation(values:[]):"
    print(custom_llm.huggingface_model(model_repo_id="bigcode/starcoder").run(code))
