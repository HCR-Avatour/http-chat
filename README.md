# HTTP Chat: Local hosted LLM chat

## Description
This project contains the resources to locally deploy a Docker container that hosts a LLama 2 13b chat model onto the Nvidia Jetson Orin edge computing device. The project will deploy a chatroom API which is called upon by the [Speech_Pipeline](https://github.com/HCR-Avatour/Speech_pipeline)  repo to implement the AI assistant interface for the Avatour project.

## Person responsible
Alexandra Neagu, Harry Phillips

## Notes
Developed in Docker Container. To run: `docker compose up --build -d`
