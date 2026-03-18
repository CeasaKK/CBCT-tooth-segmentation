FROM python:3.10-slim

RUN pip install nnunetv2 nibabel SimpleITK scipy torch numpy matplotlib

WORKDIR /app
COPY . .

CMD ["python", "inference.py", "--input", "scan.nii.gz", "--output", "outputs/", "--weights", "weights/"]
