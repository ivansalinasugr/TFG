import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Datos de ejemplo (año y cantidad de artículos)
años1 = [1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
años2 = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
años3 = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
cantidad_articulos_categoria1 = [1, 0, 2, 0, 0, 2, 1, 2, 2, 0, 5, 3, 9, 4, 6, 12, 9, 5, 13, 19, 11, 15, 19, 29, 27, 31, 22, 22, 66, 26, 38, 29]
cantidad_articulos_categoria2 = [1, 1, 0, 1, 1, 2, 3, 3, 1, 3, 2, 2, 5, 1, 6, 10, 13, 11, 15, 13, 13, 52, 16, 23, 16]
cantidad_articulos_categoria3 = [2, 4, 6, 11, 8, 47, 12, 18, 18]

# Crear la figura y los ejes
fig, ax = plt.subplots()

# Graficar los datos para cada categoría
ax.plot(años1, cantidad_articulos_categoria1, label='Estimación de distancia en fotografías faciales')
ax.plot(años2, cantidad_articulos_categoria2, label='Estimación de distancia en fotografías faciales mediante IA')
ax.plot(años3, cantidad_articulos_categoria3, label='Estimación de distancia en fotografías faciales mediante deep learning')

# Agregar una leyenda
ax.legend()

# Configurar etiquetas de ejes y título
valores_x = np.append(np.arange(1992, 2024, 4), 2023)
ax.set_xticks(valores_x)
ax.set_xlabel('Año')
ax.set_ylabel('Número de publicaciones')


# Guardar la gráfica en un archivo Excel
df = pd.DataFrame({
    'Año': años1 + años2 + años3,
    'Categoria 1': cantidad_articulos_categoria1 + [0] * (len(años1) + len(años2) - len(cantidad_articulos_categoria1)),
    'Categoria 2': [0] * len(años1) + cantidad_articulos_categoria2 + [0] * (len(años3) - len(cantidad_articulos_categoria2)),
    'Categoria 3': [0] * (len(años1) + len(años2)) + cantidad_articulos_categoria3
})

df.to_excel('grafica_excel.xlsx', index=False)

