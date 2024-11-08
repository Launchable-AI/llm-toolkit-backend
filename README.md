# ChatGPT/LLM Toolkit Backend

This repo contains the code that allows you to host your own backend for the ChatGPT/LLM Toolikit Bubble plugin.

To setup your own server, you'll need to setup a webserver (e.g., Nginx or Apache) that can accept HTTPS traffic and route it to two docker containers.

The file "server_files/chatgpt-bubble-plugin.nginx" is a sample nginx config file that you can use to setup a server.

To run the backend, you'll need docker compose (https://docs.docker.com/compose/).  This will allow you to spin up the 2 services needed (for streaming and data processing).

## Overview of setup

1. Launch a server (on AWS, GCP, Linode, etc.)

2. Clone this repository to server:

```bash
git clone https://github.com/Launchable-AI/llm-toolkit-backend
```

3. Make sure Docker and Docker Compose are installed.  See: https://docs.docker.com/engine/install/

4. Build and run Docker containers:

```bash
sudo docker compose up --build
```

5. Set up your Data Container (the plugin element within Bubble) to point to your server.

6. Test!

## Learn More

If you'd like an in-depth guide to all of these steps (launching a server, configuring domain names, etc), please consider taking the mini-course "Self-Hosting LLM Toolkit" from Launchable Academy, which you can sign up for here: https://launchable.ai/course_bubble_ai_plugins
