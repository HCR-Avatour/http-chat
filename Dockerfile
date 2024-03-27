FROM dustynv/local_llm:r36.2.0

RUN mkdir -p /opt/local_llm/local_llm/http_chat/
COPY http_chat.py /opt/local_llm/local_llm/http_chat/__main__.py
COPY system-prompt.txt /system-prompt.txt
