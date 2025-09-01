# Use official Miniconda base image
FROM continuumio/miniconda3:latest

WORKDIR /app
COPY MTV2D.yml .
RUN conda env create -f MTV2D.yml
SHELL ["conda", "run", "-n", "MTV2D", "/bin/bash", "-c"]
RUN conda clean --all --yes
CMD ["conda", "run", "-n", "MTV2D", "python", "app.py"]
