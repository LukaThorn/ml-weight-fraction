FROM python:3.7.10-buster

ENV LISTEN_IP=* \
    LISTEN_PORT=8888
ENV LC_ALL=C

RUN curl -fsSL 'https://raw.githubusercontent.com/nodesource/distributions/1e1ee93cffc254f9fa71ca10465585496a7d148d/deb/setup_lts.x' \
    | bash - \
    && apt-get -y update \
    && apt-get -y install nodejs \
    && apt-get clean

RUN curl -fsSL -o /tmp/source-sans.zip 'https://github.com/adobe-fonts/source-sans/releases/download/3.028R/source-sans-3v028R.zip' \
    && unzip -d /tmp/source-sans /tmp/source-sans.zip \
    && cp -r /tmp/source-sans/TTF /usr/share/fonts/truetype/source-sans \
    && rm -rf /tmp/source-sans /tmp/source-sans.zip \
    && fc-cache -svf

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir poetry

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"


