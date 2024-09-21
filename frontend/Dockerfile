# Étape 1 : Utilisation de l'image Node.js officielle
FROM node:18-alpine

# Étape 2 : Créer un répertoire de travail
WORKDIR /app

# Étape 3 : Copier package.json et package-lock.json (si disponible)
COPY package.json package-lock.json* ./

# Étape 4 : Installer les dépendances
RUN npm install

# Étape 5 : Copier le reste des fichiers de l'application
COPY . .

# Étape 6 : Construire l'application
RUN npm run build

# Étape 7 : Exposer le port de l'application
EXPOSE 3000

# Étape 8 : Lancer l'application
CMD ["npm", "start"]
