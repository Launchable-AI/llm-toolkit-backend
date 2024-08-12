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

4. Change directories into the repo folder you just downloaded

```bash
cd llm-toolkit-backend
```

5. Build and run Docker containers:

```bash
sudo docker compose up --build
```
Note that if you want to run these in the background (without terminal output, and so that you can close your terminal session without killing your containers), add the -d flag (for daemon, which means background process).

```bash
sudo docker compose up --build -d
```

6. Set up your Data Container (the plugin element within Bubble) to point to your server.

7. Test!  Open your Bubble app and test your setup.

## Updating

Updating your installation when the code is updated is simple, just follow these steps.  Make sure you are in the plugin directory for all of these commands.

1. Pull the latest code.

```bash
git pull
```

2. Shutdown your running containers.

```bash
sudo docker compose down
```

3. Build the new containers.

```bash
sudo docker compose up --build -d
```

Again, you can leave off the -d flag if you want to see the output in your terminal.

## Learn More

If you'd like an in-depth guide to all of these steps (launching a server, configuring domain names, etc), please consider taking the mini-course "Self-Hosting LLM Toolkit" from Launchable Academy, which you can sign up for here: (coming soon)
