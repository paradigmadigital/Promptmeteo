FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Update apt
RUN apt-get upgrade && apt-get update
RUN apt-get install nano make -y

# Install git large files
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs -y

# Install hugging faces models
WORKDIR /home/models
RUN git lfs install
RUN git lfs clone https://huggingface.co/google/flan-t5-small
RUN git lfs clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#RUN git lfs clone https://huggingface.co/hiiamsid/sentence_similarity_spanish_es
#RUN git lfs clone https://huggingface.co/tiiuae/falcon-7b-instruct

# Copy project
WORKDIR /home
COPY ./ /home/
RUN make setup
