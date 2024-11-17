from ultralytics import YOLO
from ultralytics import settings

settings.update({'datasets_dir': './'})
model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

if __name__ == '__main__':
    # Train the model
    results = model.train(data='./datasets/yolo.yaml', epochs=100, imgsz=512, rect=True)
    # model("D:\\南京大学\\毕业设计\\labelme2YoloV8-segment\\datasets\\images\\train\\GOLD_L1D_CHA_NI1_1356_2018_293_23_10_v05_r01_c05.png")



# pip --proxy 127.0.0.1:7890 install --upgrade --cache-dir=D:\\pip\\tmp --target D:\\pip\\Lib\\site-packages torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html --trusted-host pypi.org --trusted-host download.pytorch.org --trusted-host files.pythonhosted.org

#  torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html --use-deprecated=legacy-resolver

# pip --proxy 127.0.0.1:7890 install --upgrade --cache-dir=D:\\pip\\tmp --target D:\\pip\\Lib\\site-packages gradio numba pyarrow pytorch-lightning scipy tensorflow tensorflow-intel
# pip uninstall gradio numba pyarrow pytorch-lightning scipy tensorflow tensorflow-intel

# pip --proxy 127.0.0.1:7890 install --upgrade --cache-dir=D:\\pip\\tmp --target D:\\pip\\Lib\\site-packages h5py
# pip install typing-extensions
# pip install wheel