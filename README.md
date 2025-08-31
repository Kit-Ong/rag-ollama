# Running LLM locally with Ollama + RAG

# What is Ollama
Just like GPT, Gemini, and Deepseek, Ollama is an LLM that we can use as an assistant.
What most distinguishes Ollama from other LLMs is that the available models can be downloaded, allowing them to run on a <b>local computer</b>

This is very useful if we want to learn how LLMs work because we have full access to the model. There are several types of models available, ranging from `Embedding`, `Vision`, `Tools`, and `Thinking`. Each of those models has its own capability You can check the [Ollama](https://ollama.com/search) website for more details.


# What is RAG
RAG stands for Retrieval Augmented Generation. As its definition suggests, RAG allows us to augment data as context when prompting an LLM. By augmenting the data, the LLM can provide us with more specific information.

To make it easier to understand, when you type “Who is Linus Torvalds?”, the LLM can easily provide an answer because it already has plenty of data about Linus Torvalds. However, if you ask the LLM “Who was your principal when you were in school?”, the LLM will start to hallucinate because it has no data related to you.

With RAG, we can create our own dataset; all the data we have can be stored, and later the LLM will use that data as context when answering. Very powerful, isn’t it?

In this article, I will explain the steps to get started with Ollama and RAG.


# Setup Ollama

## Install Ollama
Ollama has a CLI command that can be executed through a Python script. Download Olama [here](https://ollama.com/). 

### Model for embeding
Embedding is the process of converting data such as text, code, or images into vector form. Data in vector format helps us perform queries and obtain more accurate results.

> Was trained with no overlap of the MTEB data, which indicates that the model generalizes well across several domains, tasks and text length.

For detail, you can checkout [here](https://ollama.com/library/mxbai-embed-large)
```
mxbai-embed-large
```
### Model for code
After we perform embedding using the `mxbai-embed-large model`, we need to run queries with another model. `CodeLlama` is a model that can be used because it is specifically designed for code.

> Code Llama is a model for generating and discussing code, built on top of Llama 2. It’s designed to make workflows faster and efficient for developers and make it easier for people to learn how to code. It can generate both code and natural language about code. Code Llama supports many of the most popular programming languages used today, including Python, C++, Java, PHP, Typescript (Javascript), C#, Bash and more.

For detail, you can checkout [here](https://ollama.com/library/mxbai-embed-large)
```
codellamma
```

## Install dependencies

1. Installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.


2. ChromaDB untuk database vector

```python
pip install chromadb==0.5.0
```

3. Langchain Lib sebagai tools

```python
pip install langchain-community==0.2.3
pip install langchain==0.2.2
```

After all dependencies installed. Now we can try to run the python script to see how the magic works.
## Run 

```python
python query_ollama.py
```


