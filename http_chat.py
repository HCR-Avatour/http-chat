#!/usr/bin/env python3
import os
import logging
import threading
import numpy as np

from local_llm import Agent
from local_llm.web import WebServer
from local_llm.utils import ArgParser, print_table
from local_llm.plugins import ChatQuery, PrintStream, UserPrompt

class HttpChat(Agent):
    """
    Adds webserver to ASR/TTS voice chat agent.
    """
    def __init__(self, **kwargs):
        """
        Parameters:

          upload_dir (str) -- the path to save files uploaded from the client

        See VoiceChat and WebServer for inherited arguments.
        """
        super().__init__(**kwargs)

        self.llm = ChatQuery(**kwargs)
        self.llm.add(PrintStream(color='green', relay=True).add(self.on_eos))
        self.llm.add(self.on_llm_reply)

        self.prompt = UserPrompt(interactive=True, **kwargs)
        self.prompt.add(self.llm)

        self.pipeline = [self.prompt]

        self.server = WebServer(msg_callback=self.on_message, **kwargs)

    def on_message(self, msg, msg_type=0, metadata='', **kwargs):
        if msg_type == WebServer.MESSAGE_JSON:
            if 'chat_history_reset' in msg:
                self.llm.chat_history.reset()
                self.send_chat_history()
            if 'client_state' in msg:
                if msg['client_state'] == 'connected':
                    threading.Timer(1.0, lambda: self.send_chat_history()).start()
            if 'tts_voice' in msg:
                self.tts.voice = msg['tts_voice']
        elif msg_type == WebServer.MESSAGE_TEXT:  # chat input
            self.prompt(msg.strip('"'))
        elif msg_type == WebServer.MESSAGE_IMAGE:
            logging.info(f"recieved {metadata} image message {msg.size} -> {msg.filename}")
            self.llm.chat_history.reset()
            self.llm.chat_history.append(role='user', image=msg)
            self.send_chat_history()
        else:
            logging.warning(f"WebChat agent ignoring websocket message with unknown type={msg_type}")

    def on_llm_reply(self, text):
        self.send_chat_history()

    def on_tts_samples(self, audio):
        self.server.send_message(audio, type=WebServer.MESSAGE_AUDIO)

    def send_chat_history(self, history=None):
        # TODO convert images to filenames
        # TODO sanitize text for HTML
        if history is None:
            history = self.llm.chat_history

        history = history.to_list()

        def web_text(text):
            text = text.strip()
            text = text.strip('\n')
            text = text.replace('\n', '<br/>')
            text = text.replace('<s>', '')
            text = text.replace('</s>', '')
            return text

        def web_image(image):
            if not isinstance(image, str):
                if not hasattr(image, 'filename'):
                    return None
                image = image.filename
            return os.path.join(self.server.upload_route, os.path.basename(image))

        for entry in history:
            if 'text' in entry:
                entry['text'] = web_text(entry['text'])
            if 'image' in entry:
                entry['image'] = web_image(entry['image'])

        self.server.send_message({'chat_history': history})

    def on_eos(self, text):
        if text.endswith('</s>'):
            print_table(self.llm.model.stats)

    def start(self):
        super().start()
        self.server.start()
        return self


if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['web'])
    args = parser.parse_args()

    agent = HttpChat(**vars(args)).run()

