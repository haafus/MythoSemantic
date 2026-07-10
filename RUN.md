# Running Instructions

This script automatically sets up the working environment (downloads the necessary databases) and launches the user interface.

## 1. Preparation

Make sure you have **Python** installed on your computer (version 3.8 or higher is recommended). 

To avoid conflicts with other projects and system-wide packages, it is highly recommended to use a virtual environment. Open a terminal in the project root folder and follow these steps:

### Step 1: Create a virtual environment
```bash
python -m venv venv
```

### Step 2: Activate the virtual environment
* **Windows (Command Prompt / PowerShell):**
  ```bash
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
*(You should see `(venv)` appear at the beginning of your terminal prompt, indicating that the environment is active).*

### Step 3: Install dependencies
Once the virtual environment is activated, install the required project packages:
```bash
pip install -r requirements.txt
```

## 2. Running the Application

To launch the user interface, make sure your virtual environment is still active and run the main script:

```bash
python main.py
```

### What happens next:
1. The script will automatically check for the required folders (`chroma_db` and `cache`) in the project root.
2. If they are missing, it will automatically download the necessary databases from Google Drive and extract them.
3. Once the folders are ready, it will start the local UI server. 
4. In your terminal, you will see a message indicating that the server is running (e.g., `Uvicorn running on http://127.0.0.1:8000`). **Click on this link** (or copy and paste it into your browser) to open the project interface.

## 3. Stopping the Server
To stop the server and exit the application, simply press **`CTRL+C`** in the terminal where the server is running.