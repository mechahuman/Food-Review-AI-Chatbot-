# BhojanAI – Your Foodie Guide for Ahmedabad  

BhojanAI is an **AI-powered foodie assistant** built to help people discover the best restaurants in Ahmedabad in a fun and engaging way.  
Instead of scrolling through endless reviews, you can simply **ask BhojanAI questions in natural language** like:  

- *“Where can I get nice pasta under ₹500?”*  
- *“Show me 5 restaurants with good ambience for dinner.”*  
- *“Does Pizzaiiolo have a menu?”*  

…and BhojanAI will give you foodie-style answers with ratings, must-try dishes, and even share menus when available.  

---

## Features  
- **Conversational Q&A** – Ask in plain language and get human-like, foodie-style replies.  
- **Grounded in Real Data** – Uses a structured dataset of restaurants in Ahmedabad (ratings, cuisine, pricing, must-try dishes, menus).  
- **Menu Access** – Opens menu PDFs/links when available, or politely tells you if not.  
- **Handles Typos** – Can recognize restaurants even with small spelling mistakes (e.g., *Pizziolo → Pizzaiiolo*).  
- **Engaging Responses** – Adds emojis and descriptive foodie tone to make answers fun.  

---

## Tech Stack  
- **[LangChain](https://www.langchain.com/)** – For connecting the AI model, memory, and retrieval pipeline.  
- **[Ollama](https://ollama.ai/)** – To run open-source LLMs locally (Gemma, Mistral, etc.).  
- **[ChromaDB](https://www.trychroma.com/)** – Vector database to store and search restaurant embeddings.  
- **Sentence-Transformers CrossEncoder** – For reranking results and improving accuracy.  
- **Python** – Backend logic.  
- **Frontend (Planned)** – Interactive chat UI, deployable via Vercel/Streamlit/Next.js.  

---

## How to Run
- INSTALL REQUIRED DEPENDENCIES
- Run vectordb.py (this creates a vectordb folder containing vectorized data)
- Run app.py
- The frontend will be live at port 5000 (http://127.0.0.1:5000).

## Problem We’re Solving  
Finding where to eat in Ahmedabad can be confusing:  
- Too much information, scattered reviews  
- No conversational way to explore food options  
- Menus not always available online  

-> BhojanAI solves this by being your **friendly foodie buddy**, giving you clear, fun, and accurate suggestions grounded in real data.  

---

## Scope  
- Add a web-based chat UI (like ChatGPT-style interface).  
- Deploy online so anyone can access BhojanAI easily.  
- Expand to other cities with more datasets.  

---

## License  
This project is released under the **Apache 2.0** 

---

***To run the backend you will have to download all the git files and their required dependencies.***

*THIS IS A BETA VERSION*
*This project was refined with the assistance of LLMs*
