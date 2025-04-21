# Youtube package generator

> Use discord threads as a message queue to communicate with agents that can generate thumbnails, titles hooks and scripts for youtube videos on local through an inference manager.

## High Level Objective

The creation of youtube packaging takes me a lot of time. That's why I want to delegue it to agents. I want the inference to be on local on my high-end machine, but I want to communicate my feedback when I only have access to my phone (eg commuting), when my computer is off. That's the goal of this system.

## Mid Level Objective

- Script at the boot of my computer that load all feedbacks where it needs answers and send them to the corresponding agents
- Agents ask for different inferences (txt-to-txt, txt-to-image, image-to-text, image-to-image) to the inference manager
- the inference managers load into memory the corresponding model, batch process all the types of inference, load the next model etc
- agents send messages to the discord thread, closing the loop
- One agent is responsible for thumbnail generation. It has access to this tools:
    - view_image(image)-> textual description of an image
    - create_prompt(desc) -> textual prompt for image generation. It allow to convert a description from any body to a description that fit more the standards of what is passed to image generation models
    - generate_image(prompt) -> generate an image based on the prompt
    - upscale_image(image) -> upscale an image to a predefined size
- Another agent is responsible for the title generation. It has access to this tool:
    - a tool to know if a title is going to be cut or no. A title might be cut if the length is over 55 characters (if len(title) > 55 then titl...). We need this information when creating a title because we don't want titles to not be displayed
- The last agent is responsible for creating the hook, which is the first part of the script of the video. It has access to this tools:
    - a tool to count the number of words for the script, that then convert it to an approximate duration in time
- All the agents have access to this tool:
    - ask_feedback(str, List[images]|None) that is going to send back a message from the corresponding agent to the original thread, for me to give a feedback on what the agent created
- to send a message to an agent, I write @t for title agent, @m for the thumbail (m like miniature), and @h to the hook agent

## Implementation Notes

- Use pydantic for type management
- Use typing in all the functions, inputs and outputs
- Discord is only one interface. Create the service behind that to be compatible with MCP
- Docstrings all the functions. For the tools, precisly describe them, as it was for a junior developer, and give one example usage
- Use Oriented Object Programmation
- For the premade prompts, use {{example_variable_name}} to be replaced by the python program. Use xml structure by default for indicating different zones (example, all the instructions in an <instructions></instructions> tag)
- Use the naming conventions for python. I don't know them, maybe my file names and functions names are wrong, so correct them if needed
- Prefer more verbose and simplier over less code but more complex to read and to understand. Simpler is always better


## Context

### Beginning Context

The project is empty

### Ending Context

- `src/agents/abstract_agent.py` -> abstract class for the agents.
- `src/agents/thumbail_agent.py` -> agent responsible for thumbnail creation
- `src/agents/title_agent.py` -> agent responsible for title creation
- `src/agents/hook_agent.py` -> agent responsible for hook creation
- `src/tools/` -> directory containing all the tools, one file per tool. Create an abstract tool too.
- `src/discord_manager.py` -> responsible for loading the threads, sending the messages to the agents, converting the conversations to a format understandable from discord to the agent and vice versa
- `src/inference_manager.py` -> responsible to load the models in memory to do the different types of inference, and to send the queries to the model loaded for the inference and send it back to the agent
- `src/prompts.yaml` -> list of all the premade prompts

## Low Level Tasks

> Ordered from start to finish


# WIP

[minimal code inference](https://medium.com/@cpaggen/minimal-python-code-for-local-llm-inference-112782af509a)
[api ref with basic usage](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/)
[doc for chat completion](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion)
[image generation with pyhon](https://www.reddit.com/r/StableDiffusion/comments/1e348fa/driving_stable_diffusion_image_generation_from/)
[List of SDK/Library for using Stable Diffusion via Python Code](https://www.reddit.com/r/StableDiffusion/comments/1askr5f/list_of_sdklibrary_for_using_stable_diffusion_via/)


## Installation
- https://github.com/abetlen/llama-cpp-python
- https://python.langchain.com/docs/integrations/llms/llamacpp/#installation-with-windows
- https://llama-cpp-python.readthedocs.io/en/latest/

