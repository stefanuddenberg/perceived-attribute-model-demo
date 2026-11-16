# Run the below commands from this directory to build and run the Docker container:
#   docker build -t attribute-demo .
#   docker run --gpus all -p 8888:8888 -v "${PWD}:/work" attribute-demo
# Then open http://127.0.0.1:8888 and run `attribute_model_demo.ipynb`

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV PYTHONUNBUFFERED=1
ENV SG2_REPO=/opt/stylegan2-ada-pytorch
ENV SG2_PKL=/opt/stylegan2-ffhq-1024x1024.pkl

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/NVlabs/stylegan2-ada-pytorch/archive/d72cc7d041b42ec8e806021a205ed9349f87c6a4.tar.gz', '/tmp/sg2.tar.gz')" \
    && mkdir -p /opt/stylegan2-ada-pytorch \
    && tar -xzf /tmp/sg2.tar.gz -C /opt/stylegan2-ada-pytorch --strip-components=1 \
    && rm /tmp/sg2.tar.gz \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl', '/opt/stylegan2-ffhq-1024x1024.pkl')"

RUN pip install --no-cache-dir \
        "numpy==1.21.6" \
        "pandas==1.5.3" \
        "scikit-learn==1.3.2" \
        "matplotlib==3.7.3" \
        "notebook==6.5.6" \
        "ipywidgets==7.7.5" \
        "onnxruntime==1.16.3" \
        "opencv-python-headless==4.10.0.84" \
        "onnx==1.17.0" \
        "scikit-image==0.21.0" \
        "albumentations==1.3.1" \
        easydict \
        cython \
        click \
        requests \
        ninja \
    && pip install --no-cache-dir --no-build-isolation "insightface==0.7.3" \
    && python -c "import os, urllib.request, zipfile; os.makedirs('/root/.insightface/models/buffalo_l', exist_ok=True); urllib.request.urlretrieve('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip', '/tmp/buffalo_l.zip'); zipfile.ZipFile('/tmp/buffalo_l.zip').extractall('/root/.insightface/models/buffalo_l'); os.remove('/tmp/buffalo_l.zip'); assert os.path.isfile('/root/.insightface/models/buffalo_l/genderage.onnx')" \
    && jupyter nbextension enable --py widgetsnbextension --sys-prefix

RUN pip uninstall -y onnxruntime \
    && pip install --no-cache-dir "onnxruntime-gpu==1.16.3"

WORKDIR /work
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
