#!/bin/bash

# Verificar si se proporcionó un argumento (el mensaje del commit)
if [ $# -eq 0 ]; then
    echo "Por favor, proporciona un mensaje para el commit."
    exit 1
fi

# Guardar el mensaje del commit en una variable
commit_message="$1"

# Asegurarse de estar en el directorio del repositorio de Git
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "No estás dentro de un repositorio de Git."
    exit 1
fi

# Realizar el comando git add .
git add .

# Realizar el comando git commit con el mensaje proporcionado
git commit -m "$commit_message"

# Realizar el comando git push origin master
git push origin master

echo "Se ha realizado el commit y el push correctamente."

