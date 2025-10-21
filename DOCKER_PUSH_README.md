How to build and push frontend/backend images to Docker Hub

Important: Do NOT store plaintext Docker Hub passwords in repository files. Use `docker login` interactively.

1. Log in to Docker Hub (interactive):

```powershell
docker login
# enter your Docker Hub username and password when prompted
```

2. Set environment variables (or pass them to the script):

```powershell
$env:DOCKERHUB_USERNAME = "ramizraja016"
$env:TAG = "v1"
```

3. Build and tag images (PowerShell script):

```powershell
.\scripts\build_and_tag.ps1 -User ramizraja016 -Tag v1
```

This runs `docker-compose build` for frontend and backend and ensures the images are tagged as:
- ramizraja016/mlopsproject-backend:v1
- ramizraja016/mlopsproject-frontend:v1

4. Push images to Docker Hub (after docker login):

```powershell
.\scripts\push_images.ps1 -User ramizraja016 -Tag v1
```

Notes
- If you prefer bash on Linux/macOS, the equivalent commands are:

```bash
export DOCKERHUB_USERNAME=ramizraja016
export TAG=v1
docker-compose build backend frontend
# then tag and push
docker tag yourusername/mlopsproject-backend:latest $DOCKERHUB_USERNAME/mlopsproject-backend:$TAG
docker push $DOCKERHUB_USERNAME/mlopsproject-backend:$TAG
```

Security reminder: never paste real credentials into code or share them in plaintext. Use `docker login` interactively.
