FROM mambaorg/micromamba:2.0.8 AS micromamba

ENV PYTHONDONTWRITEBYTECODE=true
ARG BUILD_ENV=prod
ARG MAMBA_DOCKERFILE_ACTIVATE=1 

USER root
RUN mkdir -p /code
WORKDIR /code
COPY environment.yaml requirements.txt pyproject.toml /code/
COPY src /code/src/

RUN micromamba install --yes --name base  --file environment.yaml --verbose 

RUN if [ "$BUILD_ENV" = "dev" ]; then \
  micromamba run --name base pip install --no-cache-dir jupyterlab jupyterlab_iframe jupyter-resource-usage nodejs; \
fi

# RUN micromamba run --name base  pip-compile --all-extras --output-file=requirements.txt pyproject.toml
RUN micromamba run --name base pip install --no-cache-dir -r requirements.txt
RUN micromamba run --name base pip install --no-cache-dir /code

# Clean up
RUN micromamba clean --all --index-cache --packages --tarballs \
     --locks --trash --force-pkgs-dirs --yes \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type d -name '__pycache__' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete \
    && micromamba run --name base pip cache purge \
    && micromamba env export --name base --explicit

RUN wq-version 

FROM ubuntu:jammy

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"
ARG BUILD_ENV=prod
ARG JUPYTER_PORT="8888"

# Configure python environment.
# This should be the same prefix 
# as the environment used in micromamba
ENV PYTHON_ENV=/opt/conda
ENV DEBIAN_FRONTEND=noninteractive \ 
    SHELL=/bin/bash \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    USE_PYGEOS=0 \
    SPATIALITE_LIBRARY_PATH='mod_spatialite.so' \
    HOME="/home/${NB_USER}" \
    PATH="${PYTHON_ENV}/bin:${PATH}" \
    PROJ_DATA="${PYTHON_ENV}/share/proj" \
    BUILD_ENV=$BUILD_ENV \
    JUPYTERLAB_DIR=$PYTHON_ENV/share/jupyter/lab \
    JUPYTER_PORT=${JUPYTER_PORT}

## Non-root user creation
# Delete existing user with UID="${NB_UID}" if it exists
RUN if grep -q "${NB_UID}" /etc/passwd; then \
        userdel --remove $(id -un "${NB_UID}"); \
    fi
# Create a non-root user
RUN useradd --no-log-init --create-home --shell /bin/bash --no-user-group --gid $NB_GID --uid $NB_UID  $NB_USER \
    && chown -R $NB_UID:$NB_GID $HOME
# Copy python environment
COPY --chown=$NB_UID:$NB_GID --from=micromamba /opt/conda $PYTHON_ENV

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
        && unzip awscliv2.zip \
        && ./aws/install \ 
        && rm awscliv2.zip \
        && rm -rf ./aws/

COPY jupyter_lab_config.py  $PYTHON_ENV/etc/jupyter/

USER $NB_USER
RUN if [ "$BUILD_ENV" = "dev" ]; then \
    jupyter server extension enable --py --sys-prefix jupyterlab_iframe jupyter_resource_usage; \
    fi 

USER root
# clean up
RUN apt clean autoclean -y \
    && apt autoremove -y \
    && rm -rf /var/lib/{apt,dpkg,cache}/

# Configure entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER $NB_USER
WORKDIR $HOME
ENTRYPOINT ["/entrypoint.sh"]