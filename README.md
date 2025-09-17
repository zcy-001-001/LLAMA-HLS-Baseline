# [LLama2-7B-baseline]

## Table of Contents
1.  [Prerequisites](#-prerequisites)
    *   [Step 1: Connect to the FPGA06 Server](#step-1-connect-to-the-fpga06-server)
    *   [Step 2: Clone the Project Repository](#step-2-clone-the-project-repository)
2.  [Getting Started](#-getting-started)
    *   [Step 1: Download Model & Tokenizer](#step-1-download-model--tokenizer)
    *   [Step 2: Set Up the FPGA Environment](#step-2-set-up-the-fpga-environment)
3.  [Project Structure and File Descriptions](#-project-structure-and-file-descriptions)
4.  [Development and Testing Workflow](#-development-and-testing-workflow)
    *   [1. C Simulation (CSIM)](#1-c-simulation-csim)
    *   [2. C Synthesis (CSynth)](#2-c-synthesis-csynth)
    *   [3. Emulation (Software & Hardware)](#3-emulation-software--hardware)
    *   [4. On-Board Execution](#4-on-board-execution)

---

## 🛠️ Prerequisites

Before you begin, you need to connect to the remote server and clone the project repository.

### Step 1: Connect to the FPGA06 Server

Use SSH to connect to the server. The command varies slightly depending on your operating system.

```
For windows:
ssh connect\your email perfix@10.92.254.209 
For Linux/Mac:
ssh connect\\your email perfix@10.92.254.209 

For example, if my email is czhang539@connect.hkust-gz.edu.cn and I am using Windows, I would connect with ssh connect\czhang539@10.92.254.209.

// Initial password is your campus email password
// Please change your password after logging in(passwd)
```
### Step 2: Clone the Project Repository
Once you are logged into the server, clone the specific Tiny-llama-baseline branch of the repository with the following command:
```
git clone -b llama2-7B-Baseline git@github.com:zcy-001-001/LLAMA-HLS-Baseline.git
cd LLAMA-HLS-Baseline
```

## 🚀 Getting Started

Follow these steps to set up the project environment and run the inference on your FPGA.

### Step 1: Download Model & Tokenizer

Before running the project, you need to download the necessary llama2-7b model weights and the tokenizer.

1.  **Download the files** from the following Google Drive link:
    *   [**Download LLama2-7B-weights & tokenizer**](https://drive.google.com/drive/folders/14Dx7VaHJ8Rb8cOp7B9Yy-7Urvhpyz-5h?usp=drive_link)

2.  **Place the files** in a directory named `llama2-7b-baseline` located in your home directory.
    *   For **Linux/macOS**, the path should be `~/llama2-7b-baseline/`.
    *   For **Windows**, the path should be `C:\Users\<YourUsername>\llama2-7b-baseline\`.

The project expects the model files to be in this specific location.

### Step 2: Set Up the FPGA Environment

Next, configure the environment for Vivado, Vitis, and the Xilinx Runtime (XRT), and verify that your FPGA board is detected by the system.

Simply run the following commands in your terminal:

1.  **Grant execution permissions** to the environment script (you only need to do this once):
    ```sh
    chmod +x env.sh
    ```

2.  **Execute the script** to load the required environment variables into your current session:
    ```sh
    source env.sh
    ```

3.  **Verify the FPGA status** to ensure it's online and recognized:
    ```sh
    xbmgmt examine
    ```
    You should see output detailing the status of your connected FPGA board.

---

## 📂 Project Structure and File Descriptions

This section outlines the structure of the project, explaining the purpose of each key directory and file.

### `./host/`
This directory contains the C++ host application code for Software Emulation (`sw_emu`), Hardware Emulation (`hw_emu`), and on-board hardware execution (`hw_test`).

*   **Functionality**: The host file defines memory allocation and controls the kernel's launch, execution, and stop.
*   **Pre-compiled Executable**: A pre-compiled executable (`llama2.exe`) is provided for convenience.
*   **Compiling**: You can compile the host code manually using the following command:
    ```sh
    g++ -g -std=c++17 -Wall -O0 ./host/llama2.cpp \
        -o ./host/llama2.exe \
        -I$XILINX_XRT/include/ -L$XILINX_XRT/lib -lxrt_coreutil -pthread
    ```
*   **⚠️ Important**:
    *   If you modify the top-level file's interface, you **must** update the host code accordingly.
    *   You need to **update the paths** to the model weights, tokenizer, and the bitstream file (`.xclbin`) within the host source code to match your local environment.

### `./hw_end1/`
This directory contains a pre-compiled bitstream file (`.xclbin`). It can be programmed directly onto the FPGA board for immediate inference. For an example of how to program the board, refer to `run_inference.sh`.

### `./llama2/`
This directory holds the configuration file for Vitis HLS synthesis.

*   **Purpose**: This file defines critical settings such as source file paths, the top-level module name, the testbench, and the target clock frequency.
*   **👉 Action Required**: You must update the source file paths in this configuration to point to your project's location.

### `./src/`
This is the most crucial directory, containing the core HLS source code for the accelerator.

*   `config.h` & `typedefs.h`: Define the model's parameters, including `dim`, `hidden_dim`, `num_layers`, etc. **No modifications are typically needed here.**
*   `forward.cpp`: The top-level file that defines the network's forward pass architecture, including the embedding layer and transformer blocks. **This is a primary file for performance optimization.**
*   `forward.h`: Defines all sub-functions used in the Llama2 model, such as quantization/de-quantization, GEMV, and various non-linear activation functions. **This is another key area for optimization.**
*   `tb_llama2_csim.cpp`: The testbench for C simulation. If you change the ports of the top-level module (`forward.cpp`), you must update this file to match.
*   `u55C1.cfg`: The Vitis linker configuration file. It connects the kernel's memory interfaces (AXI ports) to specific HBM channels. You can modify this file to **optimize memory access patterns and bandwidth.**
*   `independent_kernal/`: A subdirectory containing the source files specifically configured for software emulation (`sw_emu`).

### `./sw_emu/`
This directory contains configuration files required for the software emulation flow.

---

## ✅ Development and Testing Workflow

Follow this step-by-step guide to verify, synthesize, and deploy the accelerator.

### 1. C Simulation (CSIM)

First, verify the functional correctness of the HLS C++ code.

*   **Action**: Run the `csim.sh` script.
    ```sh
    ./csim.sh
    ```
*   **Success Condition**: The script will complete successfully, indicating that the C++ code is functionally correct. The successful functional verification is shown below:
   <img width="2914" height="346" alt="image" src="https://github.com/user-attachments/assets/87f1797c-bcc6-4ab5-8704-02fc9f8d22d8" />

### 2. C Synthesis (CSynth)

After functional verification, run C-synthesis to get a preliminary report on resources and latency.

*   **Action**: Run the `csynth.sh` script.
    ```sh
    ./csynth.sh
    ```
*   **Execution Time**: This process takes approximately 15 minutes.
*   **Output**: The script generates a `synth/` directory containing synthesis reports. We typically use the `.rpt` file as the official `csynth` report.
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/173184bb-ae72-4981-af1e-1ccfb196fac8" />
<img width="2926" height="1294" alt="image" src="https://github.com/user-attachments/assets/3d78c8c7-678e-4d95-9188-3d484c3d621a" />


### 3. Emulation (Software & Hardware)

Emulation tests the interaction between the host application and the compiled kernel without needing to build the full hardware bitstream.

*   **Action**: You can run `software_emulation.sh` or `hardware_emulation.sh`.
    ```sh
    # For Software Emulation (Recommended)
    ./software_emulation.sh

    # For Hardware Emulation
    ./hardware_emulation.sh
    ```
*   **⚠️ Note**: Hardware Emulation (`hw_emu`) is not recommended as it is significantly slower.
*   **Success Condition**: The verification passes if you see the following output:
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/7501995d-98c9-40ea-8006-f6b105ab01bd" />

### 4. On-Board Execution

This is the final step: running the accelerator on the physical FPGA board.

*   **Prerequisite**: Functionality, resource, and latency verifications from previous steps are all successful.
*   **Action**: Run the `run_inference.sh` script.You can see the video in our slides.
    ```sh
    ./run_inference.sh
    ```
*   **⏱️ Important Note**: A full compilation from source takes **4-5 hours**. A pre-built bitstream is available in `./hw_end1/` and is used by the script for immediate testing.
*   **⚙️ Configuration**: You can modify the host file (`./host/llama2.cpp`) to change modes (e.g., dialogue mode) and settings (e.g., temperature). 
 <img width="1268" height="292" alt="image" src="https://github.com/user-attachments/assets/e29c415a-3e3a-4156-bcc0-382ee52b1288" />
