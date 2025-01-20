## **Setup Instructions**

### **Prerequisites**
1. Install **Docker**:
   - Follow the installation guide for your operating system: [Get Docker](https://docs.docker.com/get-docker/).
2. Install the **Remote - Containers** extension in VSCode:
   - Open VSCode, go to the Extensions Marketplace (`Ctrl+Shift+X`), and search for `Dev Containers`.

### **Build and Start the Environment**

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Build the Docker containers:
   ```bash
   docker-compose build
   ```

3. Start the containers:
   ```bash
   docker-compose up
   ```

   This will start the containerized environment.

### **Connecting to the Running Container**

1. Open **VSCode** and connect to the running container:
   - From the VSCode Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), select:
     ```
     Dev Containers: Attach to Running Container...
     ```
   - Choose the container running your app (likely named `app` or similar).

2. Once connected, you’ll be inside the container’s workspace.

---

### **Running the Inference Script**

Once you are inside the running container:
1. Navigate to the application directory (if not already there):
   ```bash
   cd /app
   ```

2. Run the inference script:
   ```bash
   python3 inference.py
   ```

---

## **Stopping the Containers**

To stop the running containers:
1. From your terminal:
   ```bash
   docker-compose down
   ```

   This will stop and clean up the containers.

---

## **Development Workflow**

1. Make changes to your project files locally or inside the container using VSCode.
2. Save your work, and changes will automatically reflect inside the container.
3. Rebuild the container when new dependencies are added:
   ```bash
   docker-compose build
   ```
