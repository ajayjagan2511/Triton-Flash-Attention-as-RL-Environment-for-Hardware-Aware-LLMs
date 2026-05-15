FROM docker.io/library/amazonlinux:2023.8.20250818.0

ENV PM_CONTAINERIZED=1
ENV UV_VERSION=0.8.15
ENV UV_INSTALL_DIR=/opt/uv
ENV PATH="/opt/uv/:$PATH"
ENV UV_CACHE_DIR="/tmp/uv_cache"
# Share a single Python installation between all users
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV PYTHON_VERSION=3.12.11

RUN \
    dnf update -y && \
    dnf install -y \
    # Provides `groupadd` and `useradd`
    shadow-utils \
    # Provides `find`
    findutils \
    which \
    # Allows setting up firewall so model user can't access network
    iptables \
    # Install uv
    && dnf install -y curl-minimal tar gzip \
    && curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    \
    # Cleanup
    && dnf clean all \
    && rm -rf /tmp/* /var/tmp/*

# Pre-install Python so it's available for all users
RUN uv python install ${PYTHON_VERSION}

# INSTALL EXTRA SYSTEM DEPENDENCIES HERE

# Set up a user for the model.
# Tools use the demote ID to run as the model user, e.g., for executing shell commands.
ENV PM_DEMOTE_ID=1000
RUN groupadd -g ${PM_DEMOTE_ID} model && \
    useradd -m -u ${PM_DEMOTE_ID} -g ${PM_DEMOTE_ID} model && \
    gpasswd -d model root || true # Remove user `model` from group `root`

# Set up the working directory for the model
ENV MODEL_WORKDIR=/workdir
RUN mkdir -p ${MODEL_WORKDIR} && chown -R model:model ${MODEL_WORKDIR}

USER model
WORKDIR ${MODEL_WORKDIR}

# Install additional Python dependencies required for the tasks
COPY --chown=model env_requirements.txt ${MODEL_WORKDIR}/env_requirements.txt
RUN uv venv --python=${PYTHON_VERSION} \
    && uv pip install -r env_requirements.txt \
    && rm -rf ${UV_CACHE_DIR}/*

# Copy environment data
COPY --chown=model env_data/ ${MODEL_WORKDIR}/

# Set up the environment and copy the tasks into it
USER root
ENV ROOT_WORKDIR=/pm_env
WORKDIR ${ROOT_WORKDIR}

# Copy scoring data
RUN mkdir ${ROOT_WORKDIR}/scoring_data
COPY --chown=root scoring_data/ ${ROOT_WORKDIR}/scoring_data/

# Install dependencies first for better layer caching
COPY pyproject.toml ${ROOT_WORKDIR}/pyproject.toml
RUN uv sync --python=${PYTHON_VERSION} --no-install-package pm_env \
    && rm -rf ${UV_CACHE_DIR}/*

# Install environment
COPY src/ ${ROOT_WORKDIR}/src/
RUN uv pip install . \
    && rm -rf ${UV_CACHE_DIR}/*

# Configure the Python executable that will be used by the model
ENV VIRTUAL_ENV=${MODEL_WORKDIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
RUN echo -e "\033[34mModel uses $(python --version)\033[0m"

# Lock down /pm_env directory to prevent model user from accessing it
# This must be done AFTER all operations that create files in /pm_env
RUN chmod 0700 ${ROOT_WORKDIR}

RUN . ${ROOT_WORKDIR}/.venv/bin/activate && pm_env check

WORKDIR ${MODEL_WORKDIR}

# Expose NVIDIA drivers so GPU workloads can find them
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/nvidia/bin:${PATH}
