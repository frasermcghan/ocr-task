# Document OCR and Information Retrieval    

## Install Docker (GPU support recommended)
Installation instructions can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Clone this repo
```bash
git clone https://github.com/frasermcghan/ocr-task.git
cd ocr-task
```

## Copy documents
```bash
mkdir data
cp -R pdf_documents/* data/
```

## Build Docker image (this may take a few minutes)
```bash
docker build . -t ocr-task
```

## Run interactive container
```bash
docker run -it --gpus all -v .:/ocr-task ocr-task
```
## Run OCR on documents
```bash
cd /ocr-task
python extractor.py --pdf_dir data/ --text_dir outputs/text --image_dir outputs/image
```
- This will run Tesseract OCR on the documents to extract text.
- Extracted text will be saved in a .txt file with the same name as the pdf.
- Images processed by Tesseract will be saved in a .png file with the same name as the pdf.

## Process extracted text
```bash
python processor.py --text_dir outputs/text --json_save_dir outputs/json
```
- This will construct a prompt containing the extracted text and pass it to a Llama3.2 model.
- Extracted information will be saved in a .json file with the same name as the pdf.