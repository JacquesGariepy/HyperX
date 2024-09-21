# install_docker_v1.ps1

# Fonction pour créer un dossier s'il n'existe pas
function Create-Directory($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
        Write-Host "Dossier créé : $path"
    } else {
        Write-Host "Dossier existant : $path"
    }
}

# Fonction pour créer un fichier avec un contenu spécifique s'il n'existe pas
function Create-File-If-Not-Exists($path, $content) {
    if (-not (Test-Path $path)) {
        Set-Content -Path $path -Value $content
        Write-Host "Fichier créé : $path"
    } else {
        Write-Host "Fichier existant : $path"
    }
}

# Création du Dockerfile pour le backend
$backendDockerfile = @"
FROM python:3.9

WORKDIR /app

# Copy the requirements.txt file from backend folder
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other files from backend folder
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"@
Create-File-If-Not-Exists "backend\Dockerfile" $backendDockerfile

# Création du Dockerfile pour le frontend
$frontendDockerfile = @"
FROM node:16

WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install && npm list

# Copy all frontend code
COPY . .

# Build and debug
RUN ls -la
RUN npm run build -- --verbose

EXPOSE 3000

CMD ["npm", "start"]
"@
Create-File-If-Not-Exists "frontend\Dockerfile" $frontendDockerfile

# Création du fichier docker-compose.yml
$dockerCompose = @"
version: '3'
services:
  backend:
    build:
      context: ./backend  # Set backend as context
      dockerfile: Dockerfile  # Path to Dockerfile in backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      - DATABASE_URL=sqlite:///./hyperbolic_llm.db

  frontend:
    build:
      context: ./frontend  # Set frontend as context
      dockerfile: Dockerfile  # Path to Dockerfile in frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend
"@
Create-File-If-Not-Exists "docker-compose.yml" $dockerCompose

# Création du fichier .env pour le backend
$backendEnv = @"
DATABASE_URL=sqlite:///./hyperbolic_llm.db
"@
Create-File-If-Not-Exists "backend\.env" $backendEnv

# Création du fichier .env pour le frontend
$frontendEnv = @"
REACT_APP_API_URL=http://localhost:8000
"@
Create-File-If-Not-Exists "frontend\.env" $frontendEnv

# Fonction pour lancer Docker Compose
function Start-DockerCompose {
    Write-Host "Starting Docker Compose..."
    docker-compose up -d --build

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Échec du lancement de Docker Compose"
        Write-Host "Logs de Docker Compose :"
        docker-compose logs
        return
    }

    Write-Host "Docker Compose est en cours d'exécution !"
}

# Lancement de Docker Compose
Start-DockerCompose

Write-Host "L'application est maintenant en cours d'exécution !"
Write-Host "Backend accessible à l'adresse http://localhost:8000"
Write-Host "Frontend accessible à l'adresse http://localhost:3000"
