# Running LLM locally with Ollama + RAG

# What is Ollama
Sama seperti halnya GPT, Gemini, Deepseek, Ollama adalah sebuat LLM yang bisa kita gunakan untuk membantu kita sebagai asisten.
Yang paling membedakan antara Ollama dan LLM lainnya adalah, model-model yang tersedia bisa kita download sehingga bisa berjalan di <b>local computer</b>

Ini sangat baik kalau kita mau belajar cara kerja LLM karena kita mempunyai akses penuh terhadap model. 
Ada beberapa tipe model yang tersedia, mulai dari `Embedding`, `Vision`, `Tools`, dan `Thingking`
kalian bisa lihat website [Ollama](https://ollama.com/search) untuk detail nya.


# What is RAG
RAG adalah kepanjangan dari Retrieval Augmented Generation. Sama seperti definisinya, RAG membuat kita bisa mengaugmentasi data sebagai context
saat prompting ke LLM. dengan mengaugmentasi data, LLM bisa memberikan kita data yang lebih spesifik. 

Untuk lebih mudah memahaminya, ketika kalian mengetik `Siapa itu Linus Torvald?`. LLM akan dengan mudah memberikan jawaban karena sudah memilik cukup banyak data terkait Linus Torvald. Namun, bagaimana kalau anda bertanya kepada LLM `siapa kepala sekolah anda saat sekolah?`, LLM akan mulai berhalusinasi karena tidak mempunyai data terkait diri anda.

Dengan RAG kita bisa membuat data kita sendiri, semua data yang kita miliki bisa kita simpan dan Nantinya LLM akan menjadikan data itu sebagai konteks dalam menjawab. Sangat powerful bukan. 

Pada artikel ini saya akan menjelaskan langkah-langkah untuk mulai mencoba Ollama dan RAG.


# Setup Ollama

## Install Ollama
Ollama memiliki CLI command yang bisa digunakan untuk bisa dijalankan menggunakan python script. 
Download Olama [di sini](https://ollama.com/). 

### Model for embeding
Embedding adalah proses mengubah data seperti text, code, image ke dalam bentuk Vector. Data dalam format vector akan membantu kita dalam query
dan mendapatkan hasil yang lebih akurat.

> Was trained with no overlap of the MTEB data, which indicates that the model generalizes well across several domains, tasks and text length.

For detail, you can checkout [here](https://ollama.com/library/mxbai-embed-large)
```
mxbai-embed-large
```
### Model for code
Setelah kita melakukan embedding menggunakan model `mxai-embed-large`, kita perlu query dengan menggunakan model yang lain. `CodeLlamma` adalah model yang bisa digunakan karena model ini adalah model yang spesifik untuk code. 

> Code Llama is a model for generating and discussing code, built on top of Llama 2. It’s designed to make workflows faster and efficient for developers and make it easier for people to learn how to code. It can generate both code and natural language about code. Code Llama supports many of the most popular programming languages used today, including Python, C++, Java, PHP, Typescript (Javascript), C#, Bash and more.

For detail, you can checkout [here](https://ollama.com/library/mxbai-embed-large)
```
codellamma
```

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

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

## Create database

Create the Chroma DB.

```python
python create_database.py
```

Here are steps 
+ Put all files into directory
+ 

## Run 

Query the Chroma DB.

```python
python query_ollama.py
```


