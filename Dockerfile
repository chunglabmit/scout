FROM continuumio/miniconda3

RUN apt-get update && \
    apt-get -y install g++ && \
    apt-get -y install cmake

WORKDIR /scout
COPY . .
RUN pip install -r requirements.txt
RUN pip install .
RUN wget -P /scout/models/ https://www.dropbox.com/s/thj48g6klrihuw8/unet_weights3_zika.h5

ENTRYPOINT [ "scout" ]