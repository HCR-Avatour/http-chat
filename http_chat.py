#!/usr/bin/env python3
import os
import logging
import threading
import numpy as np

from flask import Flask

from local_llm import Agent
from local_llm.web import WebServer
from local_llm.utils import ArgParser, print_table
from local_llm.plugins import ChatQuery, UserPrompt

class HttpChat(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.llm = ChatQuery(**kwargs)
        self.llm.add(self.on_llm_reply)

        self.prompt = UserPrompt(interactive=True, **kwargs)
        self.prompt.add(self.llm)

        self.pipeline = [self.prompt]

        self.app = Flask(__name__)
        self.web_thread = threading.Thread(target=lambda: self.app.run(host="0.0.0.0", port=8050, debug=True, use_reloader=False), daemon=True)

        self.response = ""

    def on_message(self, msg, msg_type=0, metadata='', **kwargs):
        if msg_type == WebServer.MESSAGE_JSON:
            if 'chat_history_reset' in msg:
                self.llm.chat_history.reset()
        elif msg_type == WebServer.MESSAGE_TEXT:  # chat input
            self.prompt(msg.strip('"'))

    def on_llm_reply(self, text):
        self.response += text
        if text.endswith('</s>'):
            print_table(self.llm.model.stats)
            print("LLM SAYS: " + self.response)
            self.response = ""

    def start(self):
        super().start()
        self.web_thread.start()
        return self

if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['web'])
    args = parser.parse_args()
    agent = HttpChat(**vars(args)).run()

