from llama_cpp import Llama

QWEN_VL_PATH = 'C:/Users/ApprenTyr/.lmstudio/models/openfree/Qwen2.5-VL-32B-Instruct-Q4_K_M-GGUF/qwen2.5-vl-32b-instruct-q4_k_m.gguf'

#basic usage
def basic_usage():
    model = Llama(model_path=QWEN_VL_PATH)
    res = model("The quick brown fox jumps ", stop=["."])
    print(res["choices"][0]["text"])
    # the lazy dog


# loading a chat model
def load_chat_model():
    model = Llama(model_path=QWEN_VL_PATH, chat_format="llama-2")
    print(model.create_chat_completion(
        messages=[{
            "role": "user",
            "content": "what is the meaning of life?"
        }]
    ))

# High level API
# Below is a short example demonstrating how to use the high-level API to for basic text completion:
def high_level_api():
    llm = Llama(
        model_path=QWEN_VL_PATH,
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        # n_ctx=2048, # Uncomment to increase the context window
    )
    output = llm(
        "Q: Name the planets in the solar system? A: ", # Prompt
        max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
        stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    print(output)

    # By default llama-cpp-python generates completions in an OpenAI compatible format:
    # {
    # "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    # "object": "text_completion",
    # "created": 1679561337,
    # "model": "./models/7B/llama-model.gguf",
    # "choices": [
    #     {
    #     "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
    #     "index": 0,
    #     "logprobs": None,
    #     "finish_reason": "stop"
    #     }
    # ],
    # "usage": {
    #     "prompt_tokens": 14,
    #     "completion_tokens": 28,
    #     "total_tokens": 42
    # }
    # }

# <create_chat_completion documentation>
# create_chat_completion(messages, functions=None, function_call=None, tools=None, tool_choice=None, temperature=0.2, top_p=0.95, top_k=40, min_p=0.05, typical_p=1.0, stream=False, stop=[], seed=None, response_format=None, max_tokens=None, presence_penalty=0.0, frequency_penalty=0.0, repeat_penalty=1.0, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, model=None, logits_processor=None, grammar=None, logit_bias=None, logprobs=None, top_logprobs=None)

# Generate a chat completion from a list of messages.

# Parameters:

#     messages (List[ChatCompletionRequestMessage]) –

#     A list of messages to generate a response for.
#     functions (Optional[List[ChatCompletionFunction]], default: None ) –

#     A list of functions to use for the chat completion.
#     function_call (Optional[ChatCompletionRequestFunctionCall], default: None ) –

#     A function call to use for the chat completion.
#     tools (Optional[List[ChatCompletionTool]], default: None ) –

#     A list of tools to use for the chat completion.
#     tool_choice (Optional[ChatCompletionToolChoiceOption], default: None ) –

#     A tool choice to use for the chat completion.
#     temperature (float, default: 0.2 ) –

#     The temperature to use for sampling.
#     top_p (float, default: 0.95 ) –

#     The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
#     top_k (int, default: 40 ) –

#     The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
#     min_p (float, default: 0.05 ) –

#     The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
#     typical_p (float, default: 1.0 ) –

#     The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
#     stream (bool, default: False ) –

#     Whether to stream the results.
#     stop (Optional[Union[str, List[str]]], default: [] ) –

#     A list of strings to stop generation when encountered.
#     seed (Optional[int], default: None ) –

#     The seed to use for sampling.
#     response_format (Optional[ChatCompletionRequestResponseFormat], default: None ) –

#     The response format to use for the chat completion. Use { "type": "json_object" } to contstrain output to only valid json.
#     max_tokens (Optional[int], default: None ) –

#     The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
#     presence_penalty (float, default: 0.0 ) –

#     The penalty to apply to tokens based on their presence in the prompt.
#     frequency_penalty (float, default: 0.0 ) –

#     The penalty to apply to tokens based on their frequency in the prompt.
#     repeat_penalty (float, default: 1.0 ) –

#     The penalty to apply to repeated tokens.
#     tfs_z (float, default: 1.0 ) –

#     The tail-free sampling parameter.
#     mirostat_mode (int, default: 0 ) –

#     The mirostat sampling mode.
#     mirostat_tau (float, default: 5.0 ) –

#     The mirostat sampling tau parameter.
#     mirostat_eta (float, default: 0.1 ) –

#     The mirostat sampling eta parameter.
#     model (Optional[str], default: None ) –

#     The name to use for the model in the completion object.
#     logits_processor (Optional[LogitsProcessorList], default: None ) –

#     A list of logits processors to use.
#     grammar (Optional[LlamaGrammar], default: None ) –

#     A grammar to use.
#     logit_bias (Optional[Dict[int, float]], default: None ) –

#     A logit bias to use.

# Returns:

#     Union[CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]] –

#     Generated chat completion or a stream of chat completion chunks.
# </create_chat_completion documentation>



# function calling
# todo better example with good description
def function_calling():
    llm = Llama(model_path=QWEN_VL_PATH, chat_format="chatml-function-calling")
    llm.create_chat_completion(
        messages = [
            {
            "role": "system",
            "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"

            },
            {
            "role": "user",
            "content": "Extract Jason is 25 years old"
            }
        ],
        tools=[{
            "type": "function",
            "function": {
            "name": "UserDetail",
            "parameters": {
                "type": "object",
                "title": "UserDetail",
                "properties": {
                "name": {
                    "title": "Name",
                    "type": "string"
                },
                "age": {
                    "title": "Age",
                    "type": "integer"
                }
                },
                "required": [ "name", "age" ]
            }
            }
        }],
        tool_choice={
            "type": "function",
            "function": {
            "name": "UserDetail"
            }
        }
    )

# vision
# TODO


# main
basic_usage()
#load_chat_model()