## Prerequisites
Firstly, access the server via SSH (in compus only).

```
For windows:
ssh connect\your email perfix@10.92.254.206 
For Linux/Mac:
ssh connect\\your email perfix@10.92.254.206 

For example, if my email is czhang539@connect.hkust-gz.edu.cn and I am using Windows, I would connect with ssh connect\czhang539@10.92.254.206.

// Initial password is your campus email password
// Please change your password after logging in(passwd)
```

Next, you need to load the Vivado, Vitis, and XRT environments and check the FPGA status, run the following script.

```
chmod +x env.sh
./env.sh
xbmgmt examine
```

Then you need to download the Tiny-llama model weights and tokenizer from [this Google Drive link](https://drive.google.com/drive/folders/1FvHkoLoQnQGsKyIu0HaY4H8_4-qd7Ilb?usp=drive_link). After downloading, you must place them in the main directory named `Tiny-llama-baseline` .

