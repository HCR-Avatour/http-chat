#!/usr/bin/env python3
import os
import re
import sys
import time
import signal
import logging
import numpy as np

from threading import Lock

from flask import Flask, request

from local_llm import LocalLM, ChatHistory, ChatTemplates
from local_llm.utils import ImageExtensions, ArgParser, KeyboardInterrupt, load_prompts, print_table

# see utils/args.py for options
parser = ArgParser()

parser.add_argument("--host", type=str, default='0.0.0.0', help="host to bind to")
parser.add_argument("--port", type=int, default=80, help="port to bind to")
parser.add_argument("--dev", type=bool, default=False, help="flask development mode")

args = parser.parse_args()

prompts = load_prompts(args.prompt)
interrupt = KeyboardInterrupt()

# load model
model = LocalLM.from_pretrained(
    args.model,
    quant=args.quant,
    api=args.api,
    max_context_len=args.max_context_len,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling,
)

# create the chat history
system_prompt = open("/system-prompt.txt", "r")
chat_history = ChatHistory(model, args.chat_template, system_prompt.read())

app = Flask(__name__)

mutex = Lock()

def deEmojify(text):
    if "\"" in text:
        text.replace("\"", "")
        
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

@app.route("/")
def query():
    prompt = request.args.get('prompt')
    if prompt is None:
        return "No prompt specified"
    if prompt.replace(" ", "").lower() == "you":
        return "Please repeat."
    if "\"" in prompt:
        prompt.replace("\"", "")

    with mutex:
        # chat_history.reset()

        entry = chat_history.append(role='user', msg=prompt)
        embedding, position = chat_history.embed_chat(return_tokens=not model.has_embed)
        reply = model.generate(
            embedding,
            streaming=True,
            kv_cache=chat_history.kv_cache,
            stop_tokens=chat_history.template.stop,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        reply.output_text = deEmojify(reply.output_text)
        bot_reply = chat_history.append(role='bot', text='') # placeholder
        for token in reply:
            continue

        chat_history.kv_cache = reply.kv_cache
        bot_reply.text = reply.output_text
        
        # situation when LLM output format is broken, recall answer after reseting chat history
        if "#" in reply.output_text or ("User:" or "AI:" or "Bot:") in reply.output_text :
            chat_history.reset()
            query()

        return reply.output_text.removesuffix("</s>")


@app.route("/reset")
def reset():
    with mutex:
        chat_history.reset()
        return "OK"

app.run(host=args.host, port=args.port, debug=args.dev)
