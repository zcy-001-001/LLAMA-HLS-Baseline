
## 🚀 Getting Started

Follow these steps to set up the project environment and run the inference on your FPGA.

### Step 1: Download Model & Tokenizer

Before running the project, you need to download the necessary Tiny-llama model weights and the tokenizer.

1.  **Download the files** from the following Google Drive link:
    *   [**Download Tiny-llama-weights & tokenizer**](https://drive.google.com/drive/folders/1FvHkoLoQnQGsKyIu0HaY4H8_4-qd7Ilb?usp=drive_link)

2.  **Place the files** in a directory named `Tiny-llama-baseline` located in your home directory.

    *   For **Linux/macOS**, the path should be `~/Tiny-llama-baseline/`.
    *   For **Windows**, the path should be `C:\Users\<YourUsername>\Tiny-llama-baseline\`.

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
    ./env.sh
    ```

3.  **Verify the FPGA status** to ensure it's online and recognized:
    ```sh
    xbmgmt examine
    ```
    You should see output detailing the status of your connected FPGA board.

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

## 🛠️ Workflow Scripts

This project includes several shell scripts to automate the development and testing workflow.

*   `csim.sh`: Runs C simulation to verify the functional correctness of the C/C++ source code.
*   `csynth.sh`: Runs C synthesis using Vitis HLS to generate RTL from the C/C++ source.
*   `software_emulation.sh`: Runs the software emulation flow, which compiles the kernel for x86 and runs it with the host application.
*   `hardware_emulation.sh`: Runs the hardware emulation flow, simulating the compiled RTL in a hardware environment.
*   `hardware_implement.sh`: Launches the full hardware implementation process (synthesis, place & route) to generate the bitstream (`.xclbin`).
*   `run_inference.sh`: Executes the host application to run inference on the physical FPGA board.
