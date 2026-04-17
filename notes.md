Setup Notes:
- Install NVIDIA Container Toolkit installed
- Get cookies.txt yt-dlp --cookies-from-browser safari https://www.youtube.com/account
- make sure to set pipline_data to user owned, not root, for container access

- Include in Dockerfile, likely not needed: RUN rm -rf /app/cookies.txt && touch /app/cookies.txt && chown $USERNAME:$USERNAME /app/cookies.txt

2. Shut Down the Docker Service 
To completely stop the Docker daemon (the engine itself) on Ubuntu:
Standard Command: Run sudo systemctl stop docker.
Socket Shutdown: If you receive a warning that the service "can still be activated by docker.socket," stop the socket as well using sudo systemctl stop docker.socket.
Verify Status: Check if Docker has stopped with sudo systemctl status docker. It should show as "inactive (dead)".