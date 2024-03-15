FROM dustynv/local_llm:r36.2.0

# RUN mkdir dist/
# RUN git clone https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC \
#                                   dist/Llama-2-7b-chat-hf-q4f16_1-MLC
# RUN git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt_libs

RUN mkdir -p /opt/local_llm/local_llm/http_chat/
COPY http_chat.py /opt/local_llm/local_llm/http_chat/__main__.py
COPY system-prompt.txt /system-prompt.txt
