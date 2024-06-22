# Estimación de la distancia cámara-sujeto en fotografías faciales mediante técnicas de aprendizaje profundo

## Descripción

Este repositorio contiene los archivos y documentos relacionados con el TFG de **Iván Salinas López**. Este TFG pretende mejorar el método actual del estado del
arte en la estimación automática de la distancia cámara-sujeto en fotografías faciales.

## Estructura del Repositorio

- **Artículos**: Carpeta que contiene artículos y documentos relevantes relacionados con el proyecto.
- **FacialSCDnet+**: Contiene el código y recursos para el modelo FacialSCDnet+.
- **Memoria PDF**: Contiene la memoria del proyecto en formato PDF.
- **Procesamiento de modelos**: Scripts y herramientas para el procesamiento de modelos y la generación de imágenes a partir de modelos 3D.
- **.gitignore**: Archivo de configuración para ignorar archivos y carpetas específicas en Git.

## Instalación

Para instalar y configurar este proyecto, sigue los siguientes pasos:

1. Clona este repositorio:
   ```bash
   git clone https://github.com/ivansalinasugr/TFG.git
   ```
2. Navega a la carpeta del proyecto:
   ```bash
   cd TFG
   ```
3. Instala las dependencias necesarias:
   ```bash
   pip install -r FacialSCDnet+/requirements.txt
   ```

## Uso
Todas las ejecuciones deben ser modificadas adecuadamente para trabajar en vuestro entorno local y archivos.

1. **FacialSCDnet+**:
   - Para entrenar el modelo, ejecutar el archivo:
     ```bash
     ./script.sh
     ```

2. **Procesamiento de modelos**:
   - Para preprocesar los modelos 3D, ejecutar los notebooks:
     ```bash
     find_transformation_matrix.ipynb
     apply_transforms.ipynb
     ```
     
3. **Generación de imágenes sintéticas**:
   - Para generar las imágenes sintéticas a partir de modelos 3D, ejecutar el script de Blender:
     ```bash
     FINAL_Synthetic_generator.blend
     ```

## Contacto

Para cualquier consulta o comentario, puedes contactarme a través de e.ivansalinas@go.ugr.es .
