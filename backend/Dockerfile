FROM python:3.9

WORKDIR /app

# Copier requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip
# Installer les dépendances et afficher la liste des packages installés
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt && \
    pip list

# Copier le reste des fichiers
COPY . .

# Afficher le contenu du répertoire pour vérification
RUN ls -la

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]