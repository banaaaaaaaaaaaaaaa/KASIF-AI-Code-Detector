# KÂŞİF: AI Code Detector

KÂŞİF is an AI-generated code detection system designed for student Python submissions in academic environments.  
It helps instructors distinguish between human-written Python code and ChatGPT-generated Python code in assignments, labs, and exams.

The system combines software engineering code features, CodeBERT embeddings, and a machine learning classifier. It also provides explainable results using SHAP analysis, feature-group influence, and code highlighting to support academic review.

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- scikit-learn
- CodeBERT
- Transformers
- PyTorch
- SHAP
- Radon

## How to Run the Project

### 1. Clone or download the project

```bash
git clone https://github.com/banaaaaaaaaaaaaaaa/KASIF-AI-Code-Detector.git
cd KASIF-AI-Code-Detector
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment

For Windows PowerShell:

venv\Scripts\Activate.ps1

For Windows Command Prompt:

venv\Scripts\activate

For macOS/Linux:

source venv/bin/activate
4. Install the required libraries
pip install -r requirements.txt
5. Run the project
python app.py
6. Open the project in the browser

After running the project, open the local link shown in the terminal:

http://127.0.0.1:5000