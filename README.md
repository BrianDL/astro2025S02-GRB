# astro2025S02-GRB
Proyecto de análisis de Eyecciones de Rayos Gamma (GRB). 

Curso de Introducción a la Astronomía ECFM-USAC 2025S02


## Compilando el reporte
Para compilar el reporte son necesarias las siguientes bibliotecas de latex:
- elsarticle

luego bastará con correr

```bash
bibtex grb
pdflatex grb.tex
```

## Como Administrar el Proyecto con Poetry
### Instalación de Poetry
Para instalar Poetry, sigue las instrucciones en la [documentación oficial](https://python-poetry.org/docs/#installation). O simplemente
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Instalar las dependencias de python
```bash
poetry install
```

### Usar el Entorno de Desarrollo
Finalmente, para entrar en un ambiente con todo las herramientes disponibles:
```bash
poetry shell
```

