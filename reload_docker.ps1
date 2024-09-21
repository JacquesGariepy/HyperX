# restart-docker.ps1

# Arrêter les conteneurs existants
Write-Host "Arrêt des conteneurs existants..." -ForegroundColor Yellow
docker-compose down

# Supprimer les conteneurs, les images et les volumes
#docker system prune -a --volumes
# Reconstruire les images

Write-Host "Reconstruction des images Docker..." -ForegroundColor Yellow
docker-compose build --no-cache

# Démarrer les nouveaux conteneurs
Write-Host "Démarrage des nouveaux conteneurs..." -ForegroundColor Yellow
docker-compose up -d

# Afficher les logs du frontend (facultatif, vous pouvez le commenter si vous ne voulez pas voir les logs)
Write-Host "Affichage des logs du frontend. Appuyez sur Ctrl+C pour arrêter l'affichage des logs." -ForegroundColor Green
docker-compose logs -f frontend backend

docker-compose exec backend pip check
docker-compose ps