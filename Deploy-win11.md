
# Comprehensive Setup Guide for Beginners

Tested on Windows 11, need to further figure out how to GPU instead of CPU.

1. Install Git Bash:

   a. Visit https://git-scm.com/download/win

   b. Download the 64-bit Git for Windows Setup.

   c. Run the installer, accepting default options.

2. Install Python:

   a. Open Git Bash terminal.

   b. Type `python --version` to trigger Python installation.

   c. Follow the installation prompts, ensuring to check "Add Python to PATH".

   d. Verify installation by reopening Git Bash and typing:
      ```
      python --version
      ```
      Expected output: Python 3.12.7 (or similar)

3. Install Miniconda:

   a. Visit https://docs.conda.io/en/latest/miniconda.html

   b. Download Windows 64-bit installer for Miniconda3.

   c. Run the installer.

   d. Check "Add Miniconda3 to my PATH environment variable".

4. Add Conda to System Path:

   a. Press Win + X, select "System".

   b. Click "Advanced system settings" on the right.

   c. Click "Environment Variables" at the bottom.

   d. In "System variables", find "Path", click "Edit".

   e. Click "New" and add:
      ```
      C:\ProgramData\miniconda3;C:\ProgramData\miniconda3\Scripts;C:\ProgramData\miniconda3\Library\bin
      ```
   f. Click "OK" on all windows to save changes.

5. Download the Repository:

   a. Open Git Bash.

   b. Navigate to your desired directory.

   c. Clone the repository:
      ```
      git clone https://github.com/jimmycgz/ai-image-rag.git
      ```

6. Set Up Conda Environment:

   a. In Git Bash, navigate to the cloned repository:
      ```
      cd ColPali-Query-Generator
      ```
   b. Create a new Conda environment:
      ```
      /c/ProgramData/miniconda3/Scripts/conda.exe create -n colpali python=3.12.7
      ```
   c. Activate the environment:
      ```
      source /c/ProgramData/miniconda3/Scripts/activate colpali
      ```

7. Install Dependencies:

   a. Ensure you're in the repository directory with the Conda environment activated.

   b. Install requirements:
      ```
      pip install -r requirements.txt
      ```
   c. If you encounter issues, try installing packages individually:
      ```
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      pip install accelerate einops transformers huggingface_hub[hf_transfer] gradio Pillow pydantic
      ```

8. Run the Application:

   a. In Git Bash, with the Conda environment activated, run:
      ```
      python app.py
      ```
   b. Wait for the application to start. This may take some time as it loads the model.

   c. Look for a local URL in the output (e.g., http://127.0.0.1:7860).

   d. Open this URL in your web browser to access the Gradio interface.

